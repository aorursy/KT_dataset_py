import pandas as pd
import numpy as np
import cv2
import os
import re
from sklearn.utils import shuffle
import random
# for visualize
import copy

# from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# from albumentations.pytorch.transforms import ToTensorV2, ToTensor

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
godimg = 0

DIR_INPUT = '../input'
DIR_TRAIN = os.path.join(DIR_INPUT, 'global-wheat-detection/train')
DIR_TEST = os.path.join(DIR_INPUT, 'global-wheat-detection/test')

TRAIN_CSV = os.path.join(DIR_INPUT, 'global-wheat-detection/train.csv')
SAMPLE_SUBMIT_CSV = os.path.join(DIR_INPUT, 'global-wheat-detection/sample_submission.csv')

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUBMIT_CSV)
train_df.shape

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-10:]
train_ids = image_ids[:-10]

valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, stage_flag, transforms=None):
        super().__init__()
        
        self.df = dataframe
        self.image_ids = dataframe['image_id'].unique()
        self.image_ids = shuffle(self.image_ids)
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * len(self.image_ids)
        self.img_size = 1024
        im_w = 1024
        im_h = 1024
        for i, img_id in enumerate(self.image_ids):
            records = self.df[self.df['image_id'] == img_id]
            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            label_tmp = []
            for box in boxes:
                x1, y1, x2, y2 = box
                label_tmp.append([0, x1, y1, x2, y2])
            self.labels[i] = np.array(label_tmp)
        
        self.image_dir = image_dir
        self.transforms = transforms
        
        # self.mosaic = False
        
        self.augment = True
        if stage_flag == "val":
            self.augment = False
    
    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = cv2.imread( os.path.join(DIR_TRAIN, str(self.image_ids[index])+".jpg"), cv2.IMREAD_COLOR)
        assert img is not None, 'Image Not Found ' + DIR_TRAIN
        h0, w0 = img.shape[:2]  # orig hw
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    
    def vis_show(self, img, index, labels):
        # print(img.shape)
        img_show = copy.deepcopy(img)
        for box in labels.astype(int):
            cv2.rectangle(img_show, (box[1], box[2]), (box[3], box[4]),(0,255,0), 2)
        save_path = './tmp_show_aug'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, self.image_ids[index]+'.png'), img_show)
    
    def __getitem__(self, index: int):
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        labels = self.labels[index]
        # self.vis_show(img, index, labels)
        
        if self.augment:
            aug_methods = ["flip", "rotate","mosaic","light_contrast"]
            #print("---aug_methods num: ", len(aug_methods))
            aug_num = 5
            while aug_num > 0:
                aug_ind = random.randint(0, len(aug_methods)-1) #[0, len(aug_methods)-1]
                if aug_methods[aug_ind]=="flip":
                    img, labels = augment_flip(img, labels)
                elif aug_methods[aug_ind]=="rotate":
                    if random.randint(0,1)==1:
                        degree = 90
                    else:
                        degree = -90
                    img, labels = random_rotate(img, labels, degrees=degree)
                elif aug_methods[aug_ind]=="affine":
                    # Augment imagespace
                    degree = random.randint(-5, 5)
                    shear = random.randint(-5, 5)
                    img, labels = random_affine(img, labels, degrees=degree, translate=0, scale=0, shear=shear)
                elif aug_methods[aug_ind]=="hsv":
                    # Augment colorspace
                    # augment_hsv(img, hgain=0.0138, sgain= 0.678, vgain=0.36)
                    img = augment_hsv2(img)
                elif aug_methods[aug_ind]=="mixup":
                    # Augment mixup_image
                    index_r = random.randint(0, self.image_ids.shape[0] - 1)
                    img_r, _, _ = self.load_image(index_r)
                    labels_r = self.labels[index_r]
                    img, labels = augment_mixup(img, labels, img_r, labels_r, alpha=0.5)
                    # self.vis_show(img, index, labels)
                elif aug_methods[aug_ind]=="mosaic":
                    index_1 = random.randint(0, self.image_ids.shape[0] - 1)
                    img_1, _, _ = self.load_image(index_1)
                    labels_1 = self.labels[index_1]
                    index_2 = random.randint(0, self.image_ids.shape[0] - 1)
                    img_2, _, _ = self.load_image(index_2)
                    labels_2 = self.labels[index_2]
                    index_3 = random.randint(0, self.image_ids.shape[0] - 1)
                    img_3, _, _ = self.load_image(index_3)
                    labels_3 = self.labels[index_3]
                    img, labels = augment_mosaic(img, labels, img_1, labels_1, img_2, labels_2, img_3, labels_3)
                elif aug_methods[aug_ind]=="light_contrast":
                    img = augment_light_contrast(img)
                elif aug_methods[aug_ind]=="illumination":
                    scale = 70.0
                    normallization = True
                    img = augment_illumination(img, scale, normallization)
                aug_num-=1
        
        # if labels[:,0].shape[0] == 0:
            # print()
        #self.vis_show(img, index, labels)
        
        d = {}
        d['boxes'] = torch.from_numpy(labels[:,1:].astype(np.float32))
        d['labels'] = torch.ones((labels[:,0].shape[0],), dtype=torch.int64)
        
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img /= 255.0
        #return torch.from_numpy(img), d
        return torch.from_numpy(torch.from_numpy(img).permute(2, 0, 1).numpy().astype(np.float32) / 255.0), d
 
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    # def load_image_and_boxes(self, index):
        # image_id = self.image_ids[index]
        # image = cv2.imread( os.path.join(DIR_TRAIN, str(image_id)+".jpg"), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0
        # records = self.df[self.df['image_id'] == image_id]
        # boxes = records[['x', 'y', 'w', 'h']].values
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # return image, boxes
     
class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(os.path.join(self.image_dir, str(image_id)+".jpg"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
        
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0        

#######################################################################################
def augment_mosaic(img1, label1, img2, label2, img3, label3, img4,label4):
    h = img1.shape[0]
    w = img1.shape[1]
    cut_x = random.randint(int(0.2*w), int(0.8*w))
    cut_y = random.randint(int(0.2*h), int(0.8*h))
    
    min_area_value = 0
    min_w_value = 25
    min_h_value = 25
    
    img_mosaic = np.zeros(img1.shape, dtype = img1.dtype)
    
    # top-left
    img_mosaic[:cut_y, :cut_x] = img1[:cut_y, :cut_x]    
    labels_1 = label1.copy()
    labels_1[:, [1, 3]] = labels_1[:, [1, 3]].clip(min=1, max=cut_x)
    labels_1[:, [2, 4]] = labels_1[:, [2, 4]].clip(min=1, max=cut_y)
    
    labels_1 = labels_1.astype(np.int32)
    labels_1 = labels_1[np.where((labels_1[:,3]-labels_1[:,1])*(labels_1[:,4]-labels_1[:,2]) > min_area_value)]
    
    # bottom-right
    img_mosaic[cut_y:h, cut_x:w] = img2[cut_y:h, cut_x:w]   
    labels_2 = label2.copy()
    labels_2[:, [1, 3]] = labels_2[:, [1, 3]].clip(min=cut_x, max=w)
    labels_2[:, [2, 4]] = labels_2[:, [2, 4]].clip(min=cut_y, max=h)
    
    labels_2 = labels_2.astype(np.int32)
    labels_2 = labels_2[np.where((labels_2[:,3]-labels_2[:,1])*(labels_2[:,4]-labels_2[:,2]) > min_area_value)]
    
    labels_1 = np.append(labels_1, labels_2, axis = 0)
           
    # top-right
    img_mosaic[:cut_y, cut_x:w] = img3[:cut_y, cut_x:w]   
    labels_3 = label3.copy()
    labels_3[:, [1, 3]] = labels_3[:, [1, 3]].clip(min=cut_x, max=w)
    labels_3[:, [2, 4]] = labels_3[:, [2, 4]].clip(min=1, max=cut_y)
    
    labels_3 = labels_3.astype(np.int32)
    labels_3 = labels_3[np.where((labels_3[:,3]-labels_3[:,1])*(labels_3[:,4]-labels_3[:,2]) > min_area_value)]
    
    labels_1 = np.append(labels_1, labels_3, axis = 0)   
    
    # bottom-left
    img_mosaic[cut_y:h, 0:cut_x] = img4[cut_y:h, 0:cut_x]   
    labels_4 = label4.copy()
    labels_4[:, [1, 3]] = labels_4[:, [1, 3]].clip(min=1, max=cut_x)
    labels_4[:, [2, 4]] = labels_4[:, [2, 4]].clip(min=cut_y, max=h)
    
    labels_4 = labels_4.astype(np.int32)
    labels_4 = labels_4[np.where((labels_4[:,3]-labels_4[:,1])*(labels_4[:,4]-labels_4[:,2]) > min_area_value)]
    
    labels_1 = np.append(labels_1, labels_4, axis = 0)
    
    # find boxes need check
    label_check1 = labels_1[np.where(labels_1[:,1] == cut_x)]
    label_check2 = labels_1[np.where(labels_1[:,3] == cut_x)]
    label_check3 = labels_1[np.where(labels_1[:,2] == cut_y)]
    label_check4 = labels_1[np.where(labels_1[:,4] == cut_y)]
    
    label_check4 = np.append(label_check4, label_check3,axis = 0)
    label_check4 = np.append(label_check4, label_check2,axis = 0)
    label_check4 = np.append(label_check4, label_check1,axis = 0)
    
    label_check4 = np.array(list(set([tuple(t) for t in label_check4])))
    
    if label_check4.shape[0]:
    
        # find boxes no need check
        labels_all = labels_1.view([('', labels_1.dtype)] * labels_1.shape[1])
        label_check = label_check4.view([('', label_check4.dtype)] * label_check4.shape[1])
        label_nocheck = np.setdiff1d(labels_all, label_check).view(labels_1.dtype).reshape(-1, labels_1.shape[1])

        # filter box-w and bow-h
        label_check4 = label_check4[np.where((label_check4[:,3]-label_check4[:,1]) > min_w_value)]
        label_check4 = label_check4[np.where((label_check4[:,4]-label_check4[:,2]) > min_h_value)]
        label_check4 = label_check4[np.where((label_check4[:,3]-label_check4[:,1])/(label_check4[:,4]-label_check4[:,2]) <7)]
        label_check4 = label_check4[np.where((label_check4[:,4]-label_check4[:,2])/(label_check4[:,3]-label_check4[:,1]) <7)]

        label_nocheck = np.append(label_nocheck, label_check4, axis = 0)  
        labels_1 = label_nocheck      
    
    
    return img_mosaic, labels_1

def augment_mixup(img, labels, img_r, labels_r, alpha=0.5):
    mixup_image = alpha*img + (1-alpha) * img_r
    # print('------------------------')
    # print(labels.shape)
    # print(labels_r.shape)
    labels = np.append(labels, labels_r, axis = 0)
    # print(labels.shape)
    
    # image, boxes = dataset.load_image_and_boxes(random.randint(0, dataset.image_ids.shape[0] - 1))
    # r_image, r_boxes = dataset.load_image_and_boxes(random.randint(0, dataset.image_ids.shape[0] - 1))
    # mixup_image = alpha*image + (1-alpha)* r_image
    # for box in boxes.astype(int):
        # cv2.rectangle(image,(box[0], box[1]),(box[2],  box[3]),(0, 0, 1), 3)
        # cv2.rectangle(mixup_image,(box[0], box[1]),(box[2],  box[3]),(0, 0, 1), 3)
        
    # for box in r_boxes.astype(int):
        # cv2.rectangle(r_image,(box[0], box[1]),(box[2],  box[3]),(1, 0, 0), 3)
        # cv2.rectangle(mixup_image,(box[0], box[1]),(box[2],  box[3]),(1, 0, 0), 3)
    return mixup_image, labels
        
def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
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

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets
    
def random_rotate(img, targets=(), degrees=10, border=0):
    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=degrees, center=(img.shape[1] / 2, img.shape[0] / 2), scale=1)

    # Combined rotation matrix
    M = R
    if (border != 0) or (M != np.eye(3)).any(): 
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

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
        area = np.fabs( w * h )
        area0 = np.fabs((targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2]))
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * 1 + 1e-16) > 0.2) & (ar < 10)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

    
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def augment_hsv2(img):
    flag = False
    num_count = 0
    
    img_original = copy.deepcopy(img)
    
    while num_count<50:
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        mean_before = np.mean(hue)
        
        value_hue = random.randint(-20,20)
        value_sat = random.randint(80,95)*0.01
        value_exp = random.randint(80,95)*0.01        

        hue = (hue + value_hue).astype(img.dtype)
        sat = np.clip(sat * value_sat, 0, 255).astype(img.dtype)
        val = np.clip(val * value_exp, 0, 255).astype(img.dtype)
        
        mean_after = np.mean(hue) 
        num_count = num_count+1
        if  int(np.fabs(mean_after -  mean_before)) in range(20) and int(mean_after)<110 and int(mean_after)>15:
            img_hsv = cv2.merge((hue, sat, val)).astype(img.dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)
            flag = True
    
    if flag:
        return img
    else:
        return img_original

def augment_flip(img, labels):
    r = np.random.randint(0, 2) -1
    cv2.flip(img, r, img)
    img_h = img.shape[0]
    img_w = img.shape[1]

    if r == 0:
        labels[:,2] = img_h - labels[:,2]
        labels[:,4] = img_h - labels[:,4]
        labels = labels[:,[0,1,4,3,2]]
    elif r == -1:
        labels[:,1] = img_w - labels[:,1]
        labels[:,3] = img_w - labels[:,3]
        labels[:,2] = img_h - labels[:,2]
        labels[:,4] = img_h - labels[:,4]
        labels = labels[:,[0,3,4,1,2]]        
    
    return img, labels

def augment_light_contrast(img):
    # dst = alpha * img + beta * blank
    alpha = np.random.uniform(0, 3, 1)*0.1 + 1
    beta = random.randint(-15,15)
    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)  
    dst = np.clip(dst, 0, 255).astype(img.dtype)
    return dst

def augment_illumination(img, scale, normallization):
    height, width = img.shape[:2]
    # IMAGE_WIDTH = 512
    # IMAGE_HEIGHT = 392
    center_x = width/2
    center_y = height/2

    R = np.sqrt(center_x ** 2 + center_y ** 2) * scale

    Gauss_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            dis = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            Gauss_map[i, j] = np.exp(-0.5 * dis / R)

    mask_x = repmat(center_x, height, width)
    mask_y = repmat(center_y, height, width)

    x1 = np.arange(width)
    x_map = repmat(x1, height, 1)

    y1 = np.arange(height)
    y_map = repmat(y1, width, 1)
    y_map = np.transpose(y_map)

    Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)

    Gauss_map = np.exp(-0.5 * Gauss_map / R)

    illumination = np.zeros([height, width, 3], np.float32)

    if normallization:
        gaussian = abs(Gauss_map)
        max_gaussian = np.max(gaussian)
        min_gaussian = np.min(gaussian)
        gaussian = (gaussian - min_gaussian) / (max_gaussian - min_gaussian)

        illumination[:, :, 0] = gaussian
        illumination[:, :, 1] = gaussian
        illumination[:, :, 2] = gaussian
    else:
        illumination[:, :, 0] = Gauss_map
        illumination[:, :, 1] = Gauss_map
        illumination[:, :, 2] = Gauss_map

    illumination_img = img * illumination

    MAX = 255
    inds = np.where(
        (illumination_img[:, :, 0] > MAX) &
        (illumination_img[:, :, 1] > MAX) &
        (illumination_img[:, :, 2] > MAX))[0]
    illumination_img[inds, :] = 255

    MIN = 0
    inds = np.where(
        (illumination_img[:, :, 0] < MIN) &
        (illumination_img[:, :, 1] < MIN) &
        (illumination_img[:, :, 2] < MIN))[0]
    illumination_img[inds, :] = 0

    illumination_img = np.uint8(illumination_img)
    # img = cv2.equalizeHist(img)

    return illumination_img



#######################################################################################


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print('platform: ', device)

import random, math
# Albumentations
def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, "train", get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, "val", get_valid_transform())
test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

#train_dataset = WheatDataset(train_df, DIR_TRAIN, get_augumentation(phase='train'))

# split the dataset in train and test set
# indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    drop_last=False,
    collate_fn=collate_fn
)

print('---------------network configuration...--------------------')
########################## train faster rcnn with a specified backbone ###########
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(feature_names=[0], output_size = 7, sampling_ratio=2)
# model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool = roi_pooler)

########################## finetune from a model pretrained on COCO ###########
# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

#replace the classifier with a new one
# num_classes which is user-defined
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

WEIGHTS_FILE = os.path.join(DIR_INPUT, "fasterrcnn/fasterrcnn_resnet50_fpn_best.pth")
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))

    
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#lr_scheduler = None


print('---------------begin Training...--------------------')
num_epochs = 100 #Increase it for better results
display_interval = 50

loss_hist = Averager()
val_loss_hist = Averager()
itr = 1
least_loss = float('inf')
for epoch in range(num_epochs):
    loss_hist.reset()
    val_loss_hist.reset()
    
    for images, targets in train_data_loader:
        # print('**************************')
        # for img in images:
            # print("type(img): ", type(img))
            # print(img)
        images = list(image.to(device) for image in images)
        # for t in targets:
            # print(t)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % display_interval == 0:
            print("Iteration #%s loss: %s" %(itr, loss_value))

        itr += 1
    
    #Validation Step
    for images, targets in valid_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        val_loss_dict = model(images, targets)

        val_losses = sum(loss for loss in val_loss_dict.values())
        val_loss_value = val_losses.item()

        val_loss_hist.send(val_loss_value)

    
    # update the learning rate
    if loss_hist.value<least_loss:
        least_loss = loss_hist.value  #average loss of all iters once epoch
        ltrain=int(least_loss*1000)/1000
        print('save model epoch= %s, loss = %s' % (epoch, ltrain))
        # torch.save(model.state_dict(), 'fasterrcnn_custom_test_ep%s_loss%s.pth' % (epoch, ltrain))
        torch.save(model.state_dict(), 'best_weights.pth')        
    else:
        if lr_scheduler is not None:
            lr_scheduler.step()
            
    #if val_loss_hist.value<least_loss:
    #    least_loss = val_loss_hist.value  #average loss of all iters once epoch
    #    lval=int(least_loss*1000)/1000
    #    torch.save(model.state_dict(), 'fasterrcnn_custom_test_ep%s_loss%s.pth' % (epoch, lval))
    #    # torch.save(model.state_dict(), 'best_weights.pth')        
    #else:
    #    if lr_scheduler is not None:
    #        lr_scheduler.step()
            
    print("Epoch #%s train_loss: %s val_loss: %s" % (epoch, loss_hist.value, val_loss_hist.value))
############################## Inference #################################
print('---------------begin inference...--------------------')

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

WEIGHTS_FILE = 'best_weights.pth'
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model.eval()

detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):
        
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        
        # im_ori = cv2.imread( os.path.join(DIR_TEST, image_id+".jpg"))
        # for box in boxes:
            # cv2.rectangle(im_ori, (box[0], box[1]), (box[2]+ box[0], box[3]+box[1]), (220, 0, 0), 2)
        
        # if not os.path.exists("vis_result"):
            # os.makedirs("vis_result")
        # cv2.imwrite(os.path.join("vis_result", image_id+"_det.jpg"), im_ori)
        
        results.append(result)
    
    
    
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()
test_df.to_csv('submission.csv', index=False)
print("All is finished!")