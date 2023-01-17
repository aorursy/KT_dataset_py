!pip install -U git+https://github.com/albumentations-team/albumentations > /dev/null # no output
!pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null # no output
import os
import re
import time
import logging
import math
import random
import json
import copy
from collections import defaultdict
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import segmentation_models_pytorch as smp
import albumentations as A

from pynvml import *
nvmlInit()

try:
    from torch.cuda.amp import autocast, GradScaler
except:
    print('Current PyTorch does NOT support AMP training')
class Config:
    seed = 5468
    arch = 'timm-efficientnet-b5'
    heads = {
        'hm': 1,
        'wh': 2,
        'reg': 2}
    head_conv = 64
    reg_offset = True
    
    # Image
    data_root = '../input/global-wheat-detection/train'
    crop_size = 512
    scale = 0.
    shift = 0.
    rotate = 15.
    shear = 5.
    down_ratio = 4

    debug = False

    # loss
    hm_weight = 1
    off_weight = 1
    wh_weight = 0.1

    # train
    batch_size = 12
    base_lr = 0.75e-4
    warmup_iters = 1000
    total_epochs = 100
    stage_epochs = 100
    freeze_bn = False
    accumulate = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ema = False
    amp = True

    # logging
    output_dir = '../input/output/centernet_effnet-b5_bifpn_fold0'
    logs_dir = os.path.join(output_dir, 'logs')
    log_interval = 10

    # saving
    checkpoint = 5
    load_model = ''
    
opt = Config()
with open('../input/wheat-splits/wheat_train_0.json', 'r') as f:
    data_dict_train = json.load(f)

with open('../input/wheat-splits/wheat_valid_0.json', 'r') as f:
    data_dict_valid = json.load(f)
def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.3) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets
def multi_scale_transforms(img_size, output_sizes=[512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]):
    size = random.choice(output_sizes)
    scale = size / img_size - 1

    return A.Compose(
        [   
            A.RandomScale(scale_limit=(scale, scale), p=1)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['category_id']
        )
    )


def get_train_transforms(output_size=512):
    return A.Compose(
        [   
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.8),
            A.OneOf([
               A.GaussNoise(p=0.9),
               A.ISONoise(p=0.9)
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(num_holes=5, max_h_size=54, max_w_size=54, fill_value=114, p=0.7),
            A.RandomCrop(height=output_size, width=output_size, p=1),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0.2,
            label_fields=['category_id']
        )
    )

def get_valid_transforms(output_size=1024):
    return A.Compose(
        [
            A.Resize(height=output_size, width=output_size, p=1.0)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['category_id']
        )
    )
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def letterbox(img, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img_dict = self.data_dict[index]
    if self.load_to_ram:
      img = img_dict['image']
    else:
      file_name = img_dict['file_name']
      img_path = os.path.join(self.data_root, file_name)
      img = cv2.imread(img_path)
      assert img is not None, 'Image Not Found ' + img_path
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    labels = img_dict['labels']
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, labels, (h0, w0), img.shape[:2]


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.data_dict) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, labels0, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        if labels0.size > 0:  # Normalized xywh to pixel xyxy format
          labels = labels0.copy()
          labels[:, 1] = w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
          labels[:, 2] = h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
          labels[:, 3] = w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
          labels[:, 4] = h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=0.,
                                  translate=0.,
                                  scale=0.,
                                  shear=0.,
                                  border=self.mosaic_border)  # border to remove

    return img4, labels4


def gaussian_radius_wh(det_size, alpha):
    height, width = det_size
    h_radiuses_alpha = int(height / 2. * alpha)
    w_radiuses_alpha = int(width / 2. * alpha)
    return h_radiuses_alpha, w_radiuses_alpha

def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                      w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
class WheatDataset(data.Dataset):

    def __init__(self, opt, data_root, data_dict, img_size=1024, transforms=None, is_train=True, load_to_ram=False):
        super().__init__()

        self.num_classes = 1
        self.mean = np.array([0.315290, 0.317253, 0.214556], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.245211, 0.238036, 0.193879], dtype=np.float32).reshape(1, 1, 3)
        self.max_objs = 256

        self.opt = opt

        self.data_root = data_root
        self.data_dict = []
        for _d in tqdm(copy.deepcopy(data_dict)):
            img_dict = {
                'file_name': os.path.basename(_d['file_name']),
                'image_id': _d['id'],
                'height': _d['height'],
                'width': _d['width'],
                'labels': []
            }
            if load_to_ram:
                file_path = os.path.join(self.data_root, img_dict['file_name'])
                img_dict['image'] = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            for annot in _d['annotations']:
                bbox = annot['bbox']
                labels = [
                    annot['category_id'],
                    (bbox[0]+bbox[2]/2)/img_dict['width'],
                    (bbox[1]+bbox[3]/2)/img_dict['height'],
                    bbox[2]/img_dict['width'],
                    bbox[3]/img_dict['height']
                ] # normalized xywh
                img_dict['labels'].append(labels)
            img_dict['labels'] = np.array(img_dict['labels'], dtype=np.float32)
            
            self.data_dict.append(img_dict)

        self.img_size = img_size
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.transforms = transforms
        self.is_train = is_train
        self.load_to_ram = load_to_ram

    def __getitem__(self, index):
        if (not self.is_train) or random.random() > 0.5:
            img, labels0, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=False)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            if labels0.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = labels0.copy()
                labels[:, 1] = ratio[0] * w * (labels0[:, 1] - labels0[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (labels0[:, 2] - labels0[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (labels0[:, 1] + labels0[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (labels0[:, 2] + labels0[:, 4] / 2) + pad[1]
        else:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        if self.is_train:
            scaler = multi_scale_transforms(self.img_size)
            scaled = scaler(image=img, bboxes=labels[:, 1:5], category_id=labels[:, 0])
            img = scaled['image']
            boxes = scaled['bboxes']
            cat = scaled['category_id']

            augment_func = get_train_transforms(output_size=self.opt.crop_size)
            augmented = augment_func(**{
                'image': img,
                'bboxes': boxes,
                'category_id': cat
                })
            img = augmented['image']
            labels = np.zeros((len(augmented['category_id']), 5), dtype=np.float32)
            if len(labels) > 0:
                labels[:, 1:5] = augmented['bboxes']
            
            img, labels = random_affine(
                img,
                labels,
                degrees=self.opt.rotate,
                translate=self.opt.shift,
                scale=self.opt.scale,
                shear=self.opt.shear)
            
        num_objs = len(labels)  # number of labels
        if num_objs > 0:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        img = (img.astype(np.float32) / 255.)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        output_h = img.shape[1] // self.opt.down_ratio
        output_w = img.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes

        # generate targets
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        
        gt_det = []
        for k in range(min(num_objs, self.max_objs)):
            label = labels[k]
            bbox = label[1:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            if h > 0 and w > 0:
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                h_radius, w_radius = gaussian_radius_wh((math.ceil(h), math.ceil(w)), 0.54)
                draw_truncate_gaussian(hm[cls_id], ct_int, h_radius, w_radius)

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                        ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        
        ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

        if self.opt.debug:
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
            meta = {'gt_det': gt_det, 'labels': labels}
            ret['meta'] = meta
        return ret
    
    def __len__(self):
        return len(self.data_dict)
from torch._six import container_abcs, string_classes, int_classes

def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        res = {key: collate([d[key] for d in batch]) for key in batch[0] if key!='instance_mask'}
        if 'instance_mask' in batch[0]:
            max_obj = max([d['instance_mask'].shape[0] for d in batch])
            instance_mask = torch.zeros(len(batch),max_obj,*(batch[0]['instance_mask'].shape[1:]))
            for i in range(len(batch)):
                num_obj = batch[i]['instance_mask'].shape[0]
                instance_mask[i,:num_obj] = torch.as_tensor(batch[i]['instance_mask'])
            res.update({'instance_mask':instance_mask})
        return res
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(elem_type))
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation


    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
class FeatureMapResampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(FeatureMapResampler, self).__init__()
        if in_channels != out_channels:
            self.reduction = Conv2d(
                in_channels, out_channels, kernel_size=1,
                bias=False,
                norm=nn.BatchNorm2d(out_channels),
                activation=None
            )
        else:
            self.reduction = None

        assert stride <= 2
        self.stride = stride

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)

        if self.stride == 2:
            x = F.max_pool2d(
                x, kernel_size=self.stride + 1,
                stride=self.stride, padding=1
            )
        elif self.stride == 1:
            pass
        else:
            raise NotImplementedError()
        return x


class EncoderWithC6(nn.Module):
    def __init__(self, encoder, out_channels):
        super(EncoderWithC6, self).__init__()
        self.encoder = encoder
        self.sampler = FeatureMapResampler(encoder.out_channels[-1], out_channels, 2)
        self.out_channels = encoder.out_channels + (out_channels, )

    def forward(self, x):
        feats = self.encoder(x)
        x = feats[-1]
        feats.append(self.sampler(x))

        return feats
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):
    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)
    

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)
class SingleBiFPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, in_channels_list, out_channels):
        super(SingleBiFPN, self).__init__()
        
        self.swish = MemoryEfficientSwish()
        self.out_channels = out_channels
        # build 5-levels bifpn
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels_list) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels_list]

        num_output_connections = [0 for _ in in_channels_list]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    lateral_conv = Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        norm=nn.BatchNorm2d(out_channels)
                    )
                    self.add_module(
                        "lateral_{}_f{}".format(input_offset, feat_level), lateral_conv
                    )
            node_info.append(out_channels)
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            self.add_module(name, Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm=nn.BatchNorm2d(out_channels),
                bias=False
            ))

    def forward(self, feats):
        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_f{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode="nearest"
                        )
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = self.swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats

class BiFPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, in_channels, out_channels, num_repeats):
        super(BiFPN, self).__init__()
        
        self.out_channels = out_channels

        # build bifpn
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats):
            if i == 0:
                in_channels_list = in_channels
            else:
                in_channels_list = [out_channels] * len(in_channels)
            self.repeated_bifpn.append(SingleBiFPN(
                in_channels_list, out_channels
            ))

    def forward(self, feats):
        for bifpn in self.repeated_bifpn:
             feats = bifpn(feats)
        return feats
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x
    
class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
    
class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )
            
class BiFPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        pyramid_channels=160,
        num_repeats=3,
        segmentation_channels=64,
        merge_policy='add',
    ):
        super().__init__()
        
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 5
        encoder_channels = encoder_channels[-5:]
        
        self.bifpn = BiFPN(encoder_channels, pyramid_channels, num_repeats)
        
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3, 4]
        ])
        
        self.merge = MergeBlock(merge_policy)
    
    def forward(self, feats):
        ps = self.bifpn(feats)
        
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, ps)]
        x = self.merge(feature_pyramid)
        
        return x
class PoseFPNNet(nn.Module):
    def __init__(self, base_name, heads, head_conv=256, pyramid_channels=160, num_repeats=3):
        super(PoseFPNNet, self).__init__()

        source = 'noisy-student' if base_name[:4] == 'timm' else 'imagenet'
        print('Pretrained source {}'.format(source))
        backbone = smp.encoders.get_encoder(base_name, weights=source)
        self.encoder = EncoderWithC6(backbone, pyramid_channels)
        self.decoder = BiFPNDecoder(self.encoder.out_channels, pyramid_channels=pyramid_channels, num_repeats=num_repeats)

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        del backbone

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features[-5:])

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return [z]

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def freeze_head(self, heads):
        for head in heads:
            for p in self.__getattr__(head).parameters():
                p.requires_grad = False

    def set_mode(self, mode, is_freeze_bn=False):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        # m.weight.requires_grad = False
                        # m.bias.requires_grad   = False

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_pose_net(base_name, heads, head_conv):
    model = PoseFPNNet(base_name, heads, head_conv)
    return model
# Helper functions

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)


class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
class CtdetLoss(nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss()
    self.crit_wh = self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0

    output = outputs[-1]
    output['hm'] = _sigmoid(output['hm'])

    hm_loss += self.crit(output['hm'], batch['hm'])
    if opt.wh_weight > 0:
        wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh'])
      
    if opt.off_weight > 0:
        off_loss += self.crit_reg(
            output['reg'], batch['reg_mask'],
            batch['ind'], batch['reg'])
        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats


class ModleWithLoss(nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

def freeze_bn(module):
    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
        module.eval()

def train_one_epoch(opt, model, optimizer, scheduler, data_loader, epoch, ema=None, scaler=None):
    start_time = time.time()

    tq = enumerate(data_loader)

    total_loss = defaultdict(float)

    model.train()
    if opt.freeze_bn:
        model.apply(freeze_bn) # freeze bn
    optimizer.zero_grad()

    for batch_idx, inputs in tq:
        for k in inputs:
            if k != 'meta':
                inputs[k] = inputs[k].to(device=opt.device, non_blocking=True)
    
        if opt.amp:
            with autocast():
                output, loss, loss_stats = model(inputs)
            scaler.scale(loss).backward()
            
            # update weights
            if (batch_idx % opt.accumulate == 0) and (batch_idx > 0):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
        
        else:
            output, loss, loss_stats = model(inputs)
            loss.backward()

            # update weights
            if (batch_idx % opt.accumulate == 0) and (batch_idx > 0):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

        # logging
        for k, v in loss_stats.items():
            total_loss[k] += v.item()
        if (batch_idx % opt.log_interval == 0) and (batch_idx > 0):
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:3d}/{:3d} batches | ms/batch {:5.2f} | lr {:5.3e} | loss {:5.4f} | hm loss {:5.4f} | wh loss {:5.4f} | reg loss {:5.4f} | memory {:5.2f} MB'.format(
                epoch, batch_idx, len(data_loader),
                elapsed * 1000 / opt.log_interval, optimizer.param_groups[0]['lr'],
                total_loss['loss'] / opt.log_interval, total_loss['hm_loss'] / opt.log_interval, total_loss['wh_loss'] / opt.log_interval, total_loss['off_loss'] / opt.log_interval,
                nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).used / 1e6))
            
            total_loss = defaultdict(float)
            start_time = time.time()


def do_evaluate(opt, model, data_loader, scaler=None):
    model.eval()
    tq = enumerate(data_loader)
    total_loss = defaultdict(float)

    with torch.no_grad():
        for batch_idx, inputs in tq:
            for k in inputs:
                if k != 'meta':
                    inputs[k] = inputs[k].to(device=opt.device, non_blocking=True)

            if opt.amp:
                with autocast():
                    output, loss, loss_stats = model(inputs)
            else:
                output, loss, loss_stats = model(inputs)

            for k, v in loss_stats.items():
                total_loss[k] += v.item() * data_loader.batch_size
    
    for k, v in total_loss.items():
        total_loss[k] = v / len(data_loader.dataset)
    
    return total_loss
def main(opt):
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    
    create_logging(opt.logs_dir, 'w')

    train_dataset = WheatDataset(opt, opt.data_root, data_dict_train, img_size=1024, transforms=get_train_transforms(opt.crop_size), is_train=True, load_to_ram=True)
    train_loader = FastDataLoader(
        train_dataset, 
        opt.batch_size,
        collate_fn=collate,
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        num_workers=4)

    valid_dataset = WheatDataset(opt, opt.data_root, data_dict_valid, img_size=1024, transforms=get_valid_transforms(1024), is_train=False, load_to_ram=False)
    valid_loader = FastDataLoader(
        valid_dataset, 
        opt.batch_size,
        collate_fn=collate,
        shuffle=False, 
        drop_last=False,
        pin_memory=True,
        num_workers=4)
    
    model = ModleWithLoss(PoseFPNNet(opt.arch, opt.heads, opt.head_conv), CtdetLoss(opt)).to(opt.device)
    if opt.ema:
        print('Training with EMA')
        ema = ModelEMA(model)
    else:
        ema = None
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.base_lr)
    num_training_steps = int(opt.total_epochs * len(train_dataset) / opt.batch_size / opt.accumulate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, opt.warmup_iters, num_training_steps)
    current_epoch = 0

    if opt.load_model != '':
        # Load model weights
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_epoch = checkpoint['epoch'] + 1

    scaler = GradScaler() if opt.amp else None
    for epoch in tqdm(range(current_epoch, current_epoch + opt.stage_epochs)):

        epoch_start_time = time.time()
        train_one_epoch(opt, model, optimizer, scheduler, train_loader, epoch, ema=ema, scaler=scaler)

        if ((epoch + 1) % opt.checkpoint == 0) and (epoch > 0):
            logging.info('Saving checkpoint...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.ema.state_dict() if opt.ema else None,
                'ema_updates': ema.updates if opt.ema else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(opt.output_dir, '{:05d}.pth'.format(epoch)))

            eval_loss = do_evaluate(opt, ema.ema if opt.ema else model, valid_loader, scaler=scaler)
            logging.info('-' * 89)
            logging.info('end of epoch {:4d} | time: {:5.2f}s | val loss {:5.4f} | val hm loss {:5.4f} | val wh loss {:5.4f} | val reg loss {:5.4f} |'.format(
                epoch, (time.time() - epoch_start_time), eval_loss['loss'], eval_loss['hm_loss'], eval_loss['wh_loss'], eval_loss['off_loss']))
            logging.info('-' * 89)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(opt.output_dir, 'last.pth'))
main(opt)