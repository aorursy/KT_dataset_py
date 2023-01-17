import random

from torchvision.transforms import functional as F





class Compose(object):

    def __init__(self, transforms):

        self.transforms = transforms



    def __call__(self, image, target):

        for t in self.transforms:  

            image, target = t(image, target)

        return image, target





class ToTensor(object):

    def __call__(self, image, target):

        image = F.to_tensor(image)

        return image, target





class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):

        self.prob = prob



    def __call__(self, image, target):

        if random.random() < self.prob:

            height, width = image.shape[-2:]

            image = image.flip(-1)

            bbox = target["boxes"]



            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

            target["boxes"] = bbox

        return image, target
import collections

import PIL.ImageDraw as ImageDraw

import PIL.ImageFont as ImageFont

import numpy as np



STANDARD_COLORS = [

    'Pink', 'Green', 'SandyBrown',

    'SeaGreen',  'Silver', 'SkyBlue', 'White',

    'WhiteSmoke', 'YellowGreen'

]





def filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map, col):

    for i in range(boxes.shape[0]):

        if scores[i] > thresh:

            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple

            if classes[i] in category_index.keys():

                class_name = category_index[classes[i]]

            else:

                class_name = 'N/A'

            display_str = str(class_name)

            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))

            box_to_display_str_map[box].append(display_str)

            box_to_color_map[box] = STANDARD_COLORS[col]

        else:

            break  # Scores have been sorted





def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):

    try:

        font = ImageFont.truetype('arial.ttf', 24)

    except IOError:

        font = ImageFont.load_default()



    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]

    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)



    if top > total_display_str_height:

        text_bottom = top

    else:

        text_bottom = bottom + total_display_str_height

    for display_str in box_to_display_str_map[box][::-1]:

        text_width, text_height = font.getsize(display_str)

        margin = np.ceil(0.05 * text_height)

        draw.rectangle([(left, text_bottom - text_height - 2 * margin),

                        (left + text_width, text_bottom)], fill=color)

        draw.text((left + margin, text_bottom - text_height - margin),

                  display_str,

                  fill='black',

                  font=font)

        text_bottom -= text_height - 2 * margin





def draw_box(image, boxes, classes, scores, category_index, thresh=0.5, line_thickness=8):

    box_to_display_str_map = collections.defaultdict(list)

    box_to_color_map = collections.defaultdict(str)

    

    col = int(random.random() * len(STANDARD_COLORS))

    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map, col)



    draw = ImageDraw.Draw(image)

    im_width, im_height = image.size

    for box, color in box_to_color_map.items():

        xmin, ymin, xmax, ymax = box

        (left, right, top, bottom) = (xmin * 1, xmax * 1,

                                      ymin * 1, ymax * 1)

        draw.line([(left, top), (left, bottom), (right, bottom),

                   (right, top), (left, top)], width=line_thickness, fill=color)

        draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)

import torch

from torch.utils.data import Dataset

import os

import pandas as pd

import json





class WheatDataSet(Dataset):

    def __init__(self, root, transforms):

        self.root = os.path.join(root)



        self.img_root = os.path.join(self.root, 'train')

        self.csv_root = os.path.join(self.root, 'train.csv')



        data = pd.read_csv(self.csv_root)

        data = data.drop(['width', 'height', 'source'], axis=1)

        img_list = data['image_id'].tolist()

        object = data['bbox'].tolist()



        self.class_dict = {"wheat": 1}



        self.transforms = transforms

        img_set = []

        name_set = []

        if object is not None:

            for img, data in zip(img_list, object):

                if img not in name_set:

                    name_set.append(img)

                    img_set.append({'name': img, 'bbox': [data]})

                else:

                    idx = name_set.index(img)

                    img_set[idx]['bbox'].append(data)

        else:

            for img in img_list:

                if img not in name_set:

                    name_set.append(img)

                    img_set.append({'name': img})

        self.img_set = img_set



    def __len__(self):

        return len(self.img_set)



    def __getitem__(self, idx):

        img_path = os.path.join(self.img_root, self.img_set[idx]['name'] + '.jpg')

        image = Image.open(img_path)



        obj = [i[1:-1].split(', ') for i in self.img_set[idx]['bbox']]

        temp = []

        for bbox in obj:

            t = [float(i) for i in bbox]

            temp.append(t)

        obj = temp



        boxes = []

        for o in obj:

            xmin = o[0]

            ymin = o[1]

            xmax = xmin + o[2]

            ymax = ymin + o[3]

            boxes.append([xmin, ymin, xmax, ymax])



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = torch.ones(boxes.shape[0], dtype=torch.int64)

        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)

        image_id = torch.tensor([idx])



        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}



        if self.transforms is not None:

            image, target = self.transforms(image, target)

        

        image /= 255.0



        return image, target





from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms as ts

import random



data_transform = {

    "train": Compose([ToTensor(),

                                RandomHorizontalFlip(0.5)]),

    "val": Compose([ToTensor()])

}



# load train data set

train_data_set = WheatDataSet('../input/global-wheat-detection', data_transform["train"])

for index in random.sample(range(0, len(train_data_set)), k=1):

    img, target = train_data_set[index]

    img = ts.ToPILImage()(img)

    draw_box(img,

                 target["boxes"].numpy(),

                 target["labels"].numpy(),

                 [1 for i in range(len(target["labels"].numpy()))],

                 {1: 'Wheat'},

                 thresh=0.5,

                 line_thickness=5)

    plt.imshow(img)

    plt.show()
import torch.nn as nn

import torch

from torch import Tensor

from collections import OrderedDict

import torch.nn.functional as F



from torch.jit.annotations import List, Dict





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):

        super(Bottleneck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d



        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,

                               kernel_size=1, stride=1, bias=False)  # squeeze channels

        self.bn1 = norm_layer(out_channel)

        

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,

                               kernel_size=3, stride=stride, bias=False, padding=1)

        self.bn2 = norm_layer(out_channel)

        

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,

                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels

        self.bn3 = norm_layer(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample



    def forward(self, x):

        identity = x

        if self.downsample is not None:

            identity = self.downsample(x)



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        out += identity

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer



        self.include_top = include_top

        self.in_channel = 64



        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,

                               padding=3, bias=False)

        self.bn1 = norm_layer(self.in_channel)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

            self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



    def _make_layer(self, block, channel, block_num, stride=1):

        norm_layer = self._norm_layer

        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),

                norm_layer(channel * block.expansion))



        layers = []

        layers.append(block(self.in_channel, channel, downsample=downsample,

                            stride=stride, norm_layer=norm_layer))

        self.in_channel = channel * block.expansion



        for _ in range(1, block_num):

            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        if self.include_top:

            x = self.avgpool(x)

            x = torch.flatten(x, 1)

            x = self.fc(x)



        return x



class IntermediateLayerGetter(nn.ModuleDict):

    __annotations__ = {

        "return_layers": Dict[str, str],

    }



    def __init__(self, model, return_layers):

        if not set(return_layers).issubset([name for name, _ in model.named_children()]):

            raise ValueError("return_layers are not present in model")



        orig_return_layers = return_layers

        return_layers = {k: v for k, v in return_layers.items()}

        layers = OrderedDict()



        for name, module in model.named_children():

            layers[name] = module

            if name in return_layers:

                del return_layers[name]

            if not return_layers:

                break



        super(IntermediateLayerGetter, self).__init__(layers)

        self.return_layers = orig_return_layers



    def forward(self, x):

        out = OrderedDict()

        

        for name, module in self.named_children():

            x = module(x)

            if name in self.return_layers:

                out_name = self.return_layers[name]

                out[out_name] = x

        return out





class FeaturePyramidNetwork(nn.Module):

    



    def __init__(self, in_channels_list, out_channels, extra_blocks=None):

        super(FeaturePyramidNetwork, self).__init__()

        

        self.inner_blocks = nn.ModuleList()

        

        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:

            if in_channels == 0:

                continue

            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)

            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)

            self.inner_blocks.append(inner_block_module)

            self.layer_blocks.append(layer_block_module)



        for m in self.children():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_uniform_(m.weight, a=1)

                nn.init.constant_(m.bias, 0)



        self.extra_blocks = extra_blocks



    def get_result_from_inner_blocks(self, x, idx):

        # type: (Tensor, int)

        

        num_blocks = 0

        for m in self.inner_blocks:

            num_blocks += 1

        if idx < 0:

            idx += num_blocks

        i = 0

        out = x

        for module in self.inner_blocks:

            if i == idx:

                out = module(x)

            i += 1

        return out



    def get_result_from_layer_blocks(self, x, idx):

        # type: (Tensor, int)

        

        num_blocks = 0

        for m in self.layer_blocks:

            num_blocks += 1

        if idx < 0:

            idx += num_blocks

        i = 0

        out = x

        for module in self.layer_blocks:

            if i == idx:

                out = module(x)

            i += 1

        return out



    def forward(self, x):

        # type: (Dict[str, Tensor])



        names = list(x.keys())

        x = list(x.values())



        last_inner = self.inner_blocks[-1](x[-1])



        results = []

        

        results.append(self.layer_blocks[-1](last_inner))



        for idx in range(len(x) - 2, -1, -1):

            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)

            feat_shape = inner_lateral.shape[-2:]

            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            last_inner = inner_lateral + inner_top_down

            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))



        if self.extra_blocks is not None:

            results, names = self.extra_blocks(results, names)



        # make it back an OrderedDict

        out = OrderedDict([(k, v) for k, v in zip(names, results)])



        return out





class LastLevelMaxPool(torch.nn.Module):



    def forward(self, x, names):

        # type: (List[Tensor], List[str])

        names.append("pool")

        x.append(F.max_pool2d(x[-1], 1, 2, 0))

        return x, names





class BackboneWithFPN(nn.Module):



    def __init__(self, backbone, return_layers, in_channels_list, out_channels):

        super(BackboneWithFPN, self).__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.fpn = FeaturePyramidNetwork(

            in_channels_list=in_channels_list,

            out_channels=out_channels,

            extra_blocks=LastLevelMaxPool(),

            )

        self.out_channels = out_channels



    def forward(self, x):

        x = self.body(x)

        x = self.fpn(x)

        return x





def resnet50_fpn_backbone():

    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],

                             include_top=False)

    for name, parameter in resnet_backbone.named_parameters():

        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:

            parameter.requires_grad_(False)



    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}



    in_channels_stage2 = resnet_backbone.in_channel // 8

    in_channels_list = [

        in_channels_stage2,  # layer1 out_channel=256

        in_channels_stage2 * 2,  # layer2 out_channel=512

        in_channels_stage2 * 4,  # layer3 out_channel=1024

        in_channels_stage2 * 8,  # layer4 out_channel=2048

    ]

    out_channels = 256

    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels)





print(resnet50_fpn_backbone())
import torch

from torch.jit.annotations import Tuple

from torch import Tensor

import torch._ops

import torchvision





def nms(boxes, scores, iou_threshold):

    # type: (Tensor, Tensor, float)

    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)





def batched_nms(boxes, scores, idxs, iou_threshold):

    # type: (Tensor, Tensor, Tensor, float)

    if boxes.numel() == 0:

        return torch.empty((0,), dtype=torch.int64, device=boxes.device)



    max_coordinate = boxes.max()

    offsets = idxs.to(boxes) * (max_coordinate + 1)



    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold)

    return keep





def remove_small_boxes(boxes, min_size):

    # type: (Tensor, float)

    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    keep = (ws >= min_size) & (hs >= min_size)

    keep = keep.nonzero().squeeze(1)

    return keep





def clip_boxes_to_image(boxes, size):

    # type: (Tensor, Tuple[int, int])



    dim = boxes.dim()

    boxes_x = boxes[..., 0::2]

    boxes_y = boxes[..., 1::2]

    height, width = size



    if torchvision._is_tracing():

        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))

        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))

        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))

        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))

    else:

        boxes_x = boxes_x.clamp(min=0, max=width)

        boxes_y = boxes_y.clamp(min=0, max=width)



    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)

    return clipped_boxes.reshape(boxes.shape)





def box_area(boxes):

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])





def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)

    area2 = box_area(boxes2)



    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])

    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])



    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]



    iou = inter / (area1[:, None] + area2 - inter)

    return iou
import torch

import math

from torch.jit.annotations import List, Tuple

from torch import Tensor





def zeros_like(tensor, dtype):

    # type: (Tensor, int) -> Tensor

    return torch.zeros_like(tensor, dtype=dtype, layout=tensor.layout,

                            device=tensor.device, pin_memory=tensor.is_pinned())





# @torch.jit.script

class BalancedPositiveNegativeSampler(object):

    def __init__(self, batch_size_per_image, positive_fraction):

        # type: (int, float)

        self.batch_size_per_image = batch_size_per_image

        self.positive_fraction = positive_fraction



    def __call__(self, matched_idxs):

        # type: (List[Tensor])

        pos_idx = []

        neg_idx = []

        for matched_idxs_per_image in matched_idxs:

            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)

            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)



            num_pos = int(self.batch_size_per_image * self.positive_fraction)

            # If the number of positive samples is not enough, all positive samples will be used directly

            num_pos = min(positive.numel(), num_pos)

            

            num_neg = self.batch_size_per_image - num_pos

            # Same as positive samples

            num_neg = min(negative.numel(), num_neg)



            # Randomly select a specified number of positive and negative samples

            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]

            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]



            pos_idx_per_image = positive[perm1]

            neg_idx_per_image = negative[perm2]



            pos_idx_per_image_mask = zeros_like(

                matched_idxs_per_image, dtype=torch.uint8

            )

            neg_idx_per_image_mask = zeros_like(

                matched_idxs_per_image, dtype=torch.uint8

            )



            pos_idx_per_image_mask[pos_idx_per_image] = torch.tensor(1, dtype=torch.uint8)

            neg_idx_per_image_mask[neg_idx_per_image] = torch.tensor(1, dtype=torch.uint8)



            pos_idx.append(pos_idx_per_image_mask)

            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx





# @torch.jit.script

def encode_boxes(reference_boxes, proposals, weights):

    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor



    wx = weights[0]

    wy = weights[1]

    ww = weights[2]

    wh = weights[3]



    proposals_x1 = proposals[:, 0].unsqueeze(1)

    proposals_y1 = proposals[:, 1].unsqueeze(1)

    proposals_x2 = proposals[:, 2].unsqueeze(1)

    proposals_y2 = proposals[:, 3].unsqueeze(1)



    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)

    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)

    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)

    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)



    ex_widths = proposals_x2 - proposals_x1

    ex_heights = proposals_y2 - proposals_y1



    ex_ctr_x = proposals_x1 + 0.5 * ex_widths

    ex_ctr_y = proposals_y1 + 0.5 * ex_heights



    gt_widths = reference_boxes_x2 - reference_boxes_x1

    gt_heights = reference_boxes_y2 - reference_boxes_y1

    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths

    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights



    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths

    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights

    targets_dw = ww * torch.log(gt_widths / ex_widths)

    targets_dh = wh * torch.log(gt_heights / ex_heights)



    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    return targets





# @torch.jit.script

class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):

        # type: (Tuple[float, float, float, float], float)

        self.weights = weights

        self.bbox_xform_clip = bbox_xform_clip



    def encode(self, reference_boxes, proposals):

        # type: (List[Tensor], List[Tensor])



        """

        Calculate the expression parameter



        """



        # Count the number of positive and negative samples of each image, 

        # so as to facilitate the later splicing together and separate after processing

        boxes_per_image = [len(b) for b in reference_boxes]

        reference_boxes = torch.cat(reference_boxes, dim=0)

        proposals = torch.cat(proposals, dim=0)



        targets = self.encode_single(reference_boxes, proposals)

        return targets.split(boxes_per_image, 0)



    def encode_single(self, reference_boxes, proposals):



        dtype = reference_boxes.dtype

        device = reference_boxes.device

        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        targets = encode_boxes(reference_boxes, proposals, weights)



        return targets



    def decode(self, rel_codes, boxes):

        # type: (Tensor, List[Tensor])

        assert isinstance(boxes, (list, tuple))

        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.size(0) for b in boxes]

        concat_boxes = torch.cat(boxes, dim=0)



        box_sum = 0

        for val in boxes_per_image:

            box_sum += val

        pred_boxes = self.decode_single(

            rel_codes.reshape(box_sum, -1), concat_boxes

        )

        return pred_boxes.reshape(box_sum, -1, 4)



    def decode_single(self, rel_codes, boxes):

        boxes = boxes.to(rel_codes.dtype)



        # xmin, ymin, xmax, ymax

        widths = boxes[:, 2] - boxes[:, 0]  # The widths of anchor

        heights = boxes[:, 3] - boxes[:, 1]  # The heights of anchor

        ctr_x = boxes[:, 0] + 0.5 * widths  # Anchor center X coordinate

        ctr_y = boxes[:, 1] + 0.5 * heights  # Anchor center Y coordinate



        wx, wy, ww, wh = self.weights  # The default is 1

        dx = rel_codes[:, 0::4] / wx  # Regression parameters of the central coordinate X of anchors

        dy = rel_codes[:, 1::4] / wy  # Regression parameters of the central coordinate Y of anchors

        dw = rel_codes[:, 2::4] / ww  # Regression parameters for predicting the width of anchors

        dh = rel_codes[:, 3::4] / wh  # Regression parameters for predicting the heights of anchors



        dw = torch.clamp(dw, max=self.bbox_xform_clip)

        dh = torch.clamp(dh, max=self.bbox_xform_clip)



        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]

        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]

        pred_w = torch.exp(dw) * widths[:, None]

        pred_h = torch.exp(dh) * heights[:, None]



        # xmin

        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        # ymin

        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_h

        # xmax

        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        # ymax

        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

        return pred_boxes





# @torch.jit.script

class Matcher(object):

    BELOW_LOW_THRESHOLD = -1

    BETWEEN_THRESHOLDS = -2



    __annotations__ = {

        'BELOW_LOW_THRESHOLD': int,

        'BETWEEN_THRESHOLDS': int,

    }



    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):

        # type: (float, float, bool)

        self.BELOW_LOW_THRESHOLD = -1

        self.BETWEEN_THRESHOLDS = -2

        assert low_threshold <= high_threshold

        self.high_threshold = high_threshold

        self.low_threshold = low_threshold

        self.allow_low_quality_matches = allow_low_quality_matches



    def __call__(self, match_quality_matrix):

        """

        Calculate the maximum IOU value of anchors matching each gtboxes, and record the index

        """

        if match_quality_matrix.numel() == 0:

            # empty targets or proposals not supported during training

            if match_quality_matrix.shape[0] == 0:

                raise ValueError(

                    "No ground-truth boxes available for one of the images "

                    "during training")

            else:

                raise ValueError(

                    "No proposal boxes available for one of the images "

                    "during training")

        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.

        if self.allow_low_quality_matches:

            all_matches = matches.clone()

        else:

            all_matches = None



        below_low_threshold = matched_vals < self.low_threshold

        between_thresholds = (matched_vals >= self.low_threshold) & (

            matched_vals < self.high_threshold

        )

        

        matches[below_low_threshold] = torch.tensor(self.BELOW_LOW_THRESHOLD)  # -1



        matches[between_thresholds] = torch.tensor(self.BETWEEN_THRESHOLDS)    # -2



        if self.allow_low_quality_matches:

            assert all_matches is not None

            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)



        return matches



    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):

        # For each gt boxes, find the anchor with the largest IOU

        # highest_quality_foreach_gt is the maximum IOU value matched

        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # Find the largest anchor index of each gt box and its IOU.

        # The largest IOU matched by a gt may have multiple anchors

        gt_pred_pairs_of_highest_quality = torch.nonzero(

            match_quality_matrix == highest_quality_foreach_gt[:, None]

        )

        pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]

        # Keep the index of the anchor that matches the gt maximum IOU, even if IOU is lower than the set threshold

        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]





def smooth_l1_loss(input, target, beta: float = 1./9, size_average: bool = True):

    n = torch.abs(input - target)

    cond = n < beta

    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if size_average:

        return loss.mean()

    return loss.sum()
import torch

from torch.jit.annotations import List, Tuple

from torch import Tensor





# @torch.jit.script

class ImageList(object):

    def __init__(self, tensors, image_sizes):

        # type: (Tensor, List[Tuple[int, int]])



        self.tensors = tensors

        self.image_sizes = image_sizes



    def to(self, device):

        # type: (Device)

        cast_tensor = self.tensors.to(device)

        return ImageList(cast_tensor, self.image_sizes)
import torch

from torch import nn, Tensor

import math

from torch.jit.annotations import List, Tuple, Dict, Optional

import torchvision





class GeneralizedRCNNTransform(nn.Module):

    def __init__(self, min_size, max_size, image_mean, image_std):

        super(GeneralizedRCNNTransform, self).__init__()

        if not isinstance(min_size, (list, tuple)):

            min_size = (min_size, )

        self.min_size = min_size

        self.max_size = max_size

        self.image_mean = image_mean

        self.image_std = image_std



    def normalize(self, image):

        dtype, device = image.dtype, image.device

        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)

        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]



    def torch_choice(self, l):

        # type: (List[int])

        index = int(torch.empty(1).uniform_(0., float(len(l))).item())

        return l[index]



    def resize(self, image, target):

        # type: (Tensor, Optional[Dict[str, Tensor]])

        """

        Zoom the picture to the specified size range, and zoom bboxes information correspondingly

        Args:

            image: Input image

            target: Enter information about the picture（Include bboxes information）

        """

        # image shape is [channel, height, width]

        h, w = image.shape[-2:]

        im_shape = torch.tensor(image.shape[-2:])

        min_size = float(torch.min(im_shape))  # Gets the minimum value in height and width

        max_size = float(torch.max(im_shape))  # Gets the maximum value in height and width

        if self.training:

            size = float(self.torch_choice(self.min_size))  # Specifies the minimum side length of the input picture

        else:

            # FIXME assume for now that testing uses the largest scale

            size = float(self.min_size[-1])

        # Calculates the zoom ratio based on the specified minimum side length and the minimum edge length of the picture

        scale_factor = size / min_size  



        if max_size * scale_factor > self.max_size:

            scale_factor = self.max_size / max_size



        # image -> image[None]   [C, H, W] -> [1, C, H, W]

        image = nn.functional.interpolate(

            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False

        )[0]



        if target is None:

            return image, target



        bbox = target["boxes"]

        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])

        target["boxes"] = bbox



        return image, target



    @torch.jit.unused

    def _onnx_batch_images(self, images, size_divisible=32):

        # type: (List[Tensor], int) -> Tensor

        max_size = []

        for i in range(images[0].dim()):

            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)

            max_size.append(max_size_i)

        stride = size_divisible

        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)

        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)

        max_size = tuple(max_size)



        padded_imgs = []

        for img in images:

            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]

            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))

            padded_imgs.append(padded_img)



        return torch.stack(padded_imgs)



    def max_by_axis(self, the_list):

        # type: (List[List[int]]) -> List[int]

        maxes = the_list[0]

        for sublist in the_list[1:]:

            for index, item in enumerate(sublist):

                maxes[index] = max(maxes[index], item)

        return maxes



    def batch_images(self, images, size_divisible=32):

        # type: (List[Tensor], int)

        """

        Package a batch of images into a batch

        """

        if torchvision._is_tracing():

            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])



        stride = float(size_divisible)

        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)

        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)



        batch_shape = [len(images)] + max_size



        batched_imgs = images[0].new_full(batch_shape, 0)

        for img, pad_img in zip(images, batched_imgs):

            # copy_: Copies the elements from src into self tensor and returns self

            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs



    def postprocess(self, result, image_shapes, original_image_sizes):

        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]])

        """

        Post processing（It mainly restores bboxes to the original image scale）

        Args:

            result: list(dict), Prediction results of network, len(result) == batch_size

            image_shapes: list(torch.Size), Size after image preprocessing and scaling, len(image_shapes) == batch_size

            original_image_sizes: list(torch.Size), The original size of the image, len(original_image_sizes) == batch_size

        """

        if self.training:

            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):

            boxes = pred["boxes"]

            boxes = resize_boxes(boxes, im_s, o_im_s)  # Zoom bboxes back to the original image scale

            result[i]["boxes"] = boxes

        return result



    def __repr__(self):

        format_string = self.__class__.__name__ + '('

        _indent = '\n    '

        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)

        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,

                                                                                         self.max_size)

        format_string += '\n)'

        return format_string



    def forward(self, images, targets=None):

        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        images = [img for img in images]

        for i in range(len(images)):

            image = images[i]

            target_index = targets[i] if targets is not None else None



            if image.dim() != 3:

                raise ValueError("images is expected to be a list of 3d tensors "

                                 "of shape [C, H, W], got {}".format(image.shape))

            image = self.normalize(image)

            image, target_index = self.resize(image, target_index)

            images[i] = image

            if targets is not None and target_index is not None:

                targets[i] = target_index

        # Record the image size after resizing

        image_sizes = [img.shape[-2:] for img in images]

        images = self.batch_images(images)  # Package images into a batch

        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])



        for image_size in image_sizes:

            assert len(image_size) == 2

            image_sizes_list.append((image_size[0], image_size[1]))



        image_list = ImageList(images, image_sizes_list)

        return image_list, targets





def resize_boxes(boxes, original_size, new_size):

    # type: (Tensor, List[int], List[int]) -> Tensor

    """

    The boxes parameter is scaled according to the image scaling

    """

    ratios = [

        torch.tensor(s, dtype=torch.float32, device=boxes.device) /

        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)

        for s, s_orig in zip(new_size, original_size)

    ]

    ratios_height, ratios_width = ratios

    # Removes a tensor dimension, boxes [minibatch, 4]

    # Returns a tuple of all slices along a given dimension, already without it.

    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratios_width

    xmax = xmax * ratios_width

    ymin = ymin * ratios_height

    ymax = ymax * ratios_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
import torch

import torch.nn.functional as F

from torch import Tensor

from torch.jit.annotations import Optional, List, Dict, Tuple





def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    # type: (Tensor, Tensor, List[Tensor], List[Tensor])

    """

    Computes the loss for Faster R-CNN.

    """

    labels = torch.cat(labels, dim=0)

    regression_targets = torch.cat(regression_targets, dim=0)



    # Calculate category loss information

    classification_loss = F.cross_entropy(class_logits, labels)



    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)



    labels_pos = labels[sampled_pos_inds_subset]



    # shape=[num_proposal, num_classes]

    N, num_classes = class_logits.shape

    box_regression = box_regression.reshape(N, -1, 4)



    # Calculate bounding box loss information

    box_loss = smooth_l1_loss(

        box_regression[sampled_pos_inds_subset, labels_pos],

        regression_targets[sampled_pos_inds_subset],

        beta=1 / 9,

        size_average=False,

    ) / labels.numel()



    return classification_loss, box_loss





class ROIHeads(torch.nn.Module):

    __annotations__ = {

        'box_coder': BoxCoder,

        'proposal_matcher': Matcher,

        'fg_bg_sampler': BalancedPositiveNegativeSampler,

    }



    def __init__(self, box_roi_pool, box_head, box_predictor,

                 # Faster R-CNN training

                 fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,

                 # Faster R-CNN inference

                 score_thresh, nms_thresh, detection_per_img):

        super(ROIHeads, self).__init__()



        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(

            fg_iou_thresh,

            bg_iou_thresh,

            allow_low_quality_matches=False

        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(

            batch_size_per_image,

            positive_fraction

        )

        if bbox_reg_weights is None:

            bbox_reg_weights = (10., 10., 5., 5.)

        self.box_coder = BoxCoder(bbox_reg_weights)



        self.box_roi_pool = box_roi_pool

        self.box_head = box_head

        self.box_predictor = box_predictor



        self.score_thresh = score_thresh

        self.nms_thresh = nms_thresh

        self.detection_per_img = detection_per_img



    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):

        # type: (List[Tensor], List[Tensor], List[Tensor])

        matched_idxs = []

        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:

                device = proposals_in_image.device

                clamped_matched_idxs_in_image = torch.zeros(

                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device

                )

                labels_in_image = torch.zeros(

                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device

                )

            else:

                # Calculate IOU

                match_quality_matrix = box_iou(gt_boxes_in_image, proposals_in_image)

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)



                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                labels_in_image = labels_in_image.to(dtype=torch.int64)



                # label background (below the low threshold)

                # background

                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD

                labels_in_image[bg_inds] = torch.tensor(0)



                # Waste samples

                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2

                labels_in_image[ignore_inds] = torch.tensor(-1)



            matched_idxs.append(clamped_matched_idxs_in_image)

            labels.append(labels_in_image)

        return matched_idxs, labels



    def subsample(self, labels):

        # type: (List[Tensor])



        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_inds = []

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):

            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)

            sampled_inds.append(img_sampled_inds)

        return sampled_inds



    def add_gt_proposals(self, proposals, gt_boxes):

        # type: (List[Tensor], List[Tensor])

        proposals = [

            torch.cat((proposal, gt_box))

            for proposal, gt_box in zip(proposals, gt_boxes)

        ]

        return proposals



    def DELTEME_all(self, the_list):

        # type: (List[bool])

        for i in the_list:

            if not i:

                return False

        return True



    def check_targets(self, targets):

        # type: (Optional[List[Dict[str, Tensor]]])

        assert targets is not None

        assert self.DELTEME_all(["boxes" in t for t in targets])

        assert self.DELTEME_all(["labels" in t for t in targets])



    def select_training_samples(self, proposals, targets):

        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        

        # Check whether the target is empty

        self.check_targets(targets)

        assert targets is not None

        dtype = proposals[0].dtype

        device = proposals[0].device



        gt_boxes = [t["boxes"].to(dtype) for t in targets]

        gt_labels = [t["labels"] for t in targets]



        # Splice gt_boxes to the back of the proposal

        proposals = self.add_gt_proposals(proposals, gt_boxes)



        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)



        # Positive and negative samples are sampled according to the given number and proportion

        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []

        num_images = len(proposals)



        for img_id in range(num_images):

            # Get the positive and negative sample index of each image

            img_sampled_inds = sampled_inds[img_id]

            proposals[img_id] = proposals[img_id][img_sampled_inds]

            labels[img_id] = labels[img_id][img_sampled_inds]

            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]



            gt_boxes_in_image = gt_boxes[img_id]

            if gt_boxes_in_image.numel() == 0:

                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])



        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, labels, regression_targets



    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):

        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])

        device = class_logits.device

        num_classes = class_logits.shape[-1]



        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_boxes = self.box_coder.decode(box_regression, proposals)



        pred_scores = F.softmax(class_logits, -1)



        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)

        pred_scores_list = pred_scores.split(boxes_per_image, 0)



        all_boxes = []

        all_scores = []

        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):

            # Cut the predicted boxes information and

            # adjust the out of bounds coordinates to the image boundary

            boxes = clip_boxes_to_image(boxes, image_shape)



            # create labels for each prediction

            labels = torch.arange(num_classes, device=device)

            labels = labels.view(1, -1).expand_as(scores)



            # Remove all information with index 0 (0 represents background)

            boxes = boxes[:, 1:]

            scores = scores[:, 1:]

            labels = labels[:, 1:]



            boxes = boxes.reshape(-1, 4)

            scores = scores.reshape(-1)

            labels = labels.reshape(-1)



            # Remove low probability targets

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)

            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]



            # Remove small target

            keep = remove_small_boxes(boxes, min_size=1e-2)

            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]



            # After NMS processing is executed, 

            # the results will be sorted from large to small according to scores

            keep = batched_nms(boxes, scores, labels, self.nms_thresh)



            # Get the top k prediction targets with scores in the top

            keep = keep[:self.detection_per_img]

            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]



            all_boxes.append(boxes)

            all_scores.append(scores)

            all_labels.append(labels)

        return all_boxes, all_scores, all_labels



    def forward(self, features, proposals, image_shapes, targets=None):

        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])

        """

        Arguments:

            features (List[Tensor])

            proposals (List[Tensor[N, 4]])

            image_shapes (List[Tuple[H, W]])

            targets (List[Dict])

        """



        # Check the data type of targets

        if targets is not None:

            for t in targets:

                floating_point_types = (torch.float, torch.double, torch.half)

                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"

                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"



        if self.training:

            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        else:

            labels = None

            regression_targets = None

            matched_idxs = None



        box_features = self.box_roi_pool(features, proposals, image_shapes)

        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)



        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])

        losses = {}

        if self.training:

            assert labels is not None and regression_targets is not None

            loss_classifier, loss_box_reg = fastrcnn_loss(

                class_logits, box_regression, labels, regression_targets)

            losses = {

                "loss_classifier": loss_classifier,

                "loss_box_reg": loss_box_reg

            }

        else:

            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

            num_images = len(boxes)

            for i in range(num_images):

                result.append(

                    {

                        "boxes": boxes[i],

                        "labels": labels[i],

                        "scores": scores[i],

                    }

                )

        return result, losses
import torch

import torchvision

from torch.nn import functional as F

from torch import nn

from torch.jit.annotations import List, Optional, Dict, Tuple

from torch import Tensor





@torch.jit.unused

def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):

    # type: (Tensor, int) -> Tuple[int, int]

    from torch.onnx import operators

    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)



    pre_nms_top_n = torch.min(torch.cat(

        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),

         num_anchors), 0).to(torch.int32)).to(num_anchors.dtype)



    return num_anchors, pre_nms_top_n





class AnchorsGenerator(nn.Module):

    __annotations__ = {

        'cell_anchors': Optional[List[torch.Tensor]],

        '_cache': Dict[str, List[torch.Tensor]]

    }



    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):

        super(AnchorsGenerator, self).__init__()



        if not isinstance(sizes[0], (list, tuple)):

            sizes = tuple((s, ) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):

            aspect_ratios = (aspect_ratios, ) * len(sizes)



        assert len(sizes) == len(aspect_ratios)



        self.sizes = sizes

        self.aspect_ratios = aspect_ratios

        self.cell_anchors = None

        self._cache = {}



    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device='cpu'):

        # type: (List[int], List[float], int, Device)

        scales = torch.as_tensor(scales, dtype=dtype, device=device)

        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(aspect_ratios)

        w_ratios = 1.0 / h_ratios



        ws = (w_ratios[:, None] * scales[None, :]).view(-1)

        hs = (h_ratios[:, None] * scales[None, :]).view(-1)



        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2



        return base_anchors.round()



    def set_cell_anchors(self, dtype, device):

        # type: (int, Device) -> None

        if self.cell_anchors is not None:

            cell_anchors = self.cell_anchors

            assert cell_anchors is not None



            if cell_anchors[0].device == device:

                return



        cell_anchors = [

            self.generate_anchors(sizes, aspect_ratios, dtype, device)

            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)

        ]

        self.cell_anchors = cell_anchors



    def num_anchors_per_location(self):

        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]



    def grid_anchors(self, grid_sizes, strides):

        # type: (List[List[int]], List[List[Tensor]])

        anchors = []

        cell_anchors = self.cell_anchors

        assert cell_anchors is not None



        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):

            grid_height, grid_width = size

            stride_height, stride_width = stride

            device = base_anchors.device



            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width

            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height



            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

            shift_x = shift_x.reshape(-1)

            shift_y = shift_y.reshape(-1)



            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)



            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)

            anchors.append(shifts_anchor.reshape(-1, 4))



        return anchors



    def cached_grid_anchors(self, grid_sizes, strides):

        # type: (List[List[int]], List[List[Tensor]])

        """Cache all the calculated anchor information"""

        key = str(grid_sizes) + str(strides)



        if key in self._cache:

            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)

        self._cache[key] = anchors

        return anchors



    def forward(self, image_list, feature_maps):

        # type: (ImageList, List[Tensor])



        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])



        image_size = image_list.tensors.shape[-2:]

        dtype, device = feature_maps[0].dtype, feature_maps[0].device



        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),

                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in grid_sizes]



        # Generate anchors template

        self.set_cell_anchors(dtype, device)



        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)



        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])



        for i, (image_height, image_width) in enumerate(image_list.image_sizes):

            anchors_in_image = []

            for anchors_per_feature_map in anchors_over_all_feature_maps:

                anchors_in_image.append(anchors_per_feature_map)

            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self._cache.clear()

        return anchors





class RPNHead(nn.Module):

    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Calculate the predicted target probability

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)



        for layer in self.children():

            if isinstance(layer, nn.Conv2d):

                nn.init.normal_(layer.weight, std=0.01)

                nn.init.constant_(layer.bias, 0)



    def forward(self, x):

        # type: (List[Tensor])

        logits = []

        bbox_reg = []

        for i, feature in enumerate(x):

            t = F.relu(self.conv(feature))

            logits.append(self.cls_logits(t))

            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg





def permute_and_flatten(layer, N, A, C, H, W):

    # type: (Tensor, int, int, int, int, int)

    """Adjust the tensor order and reshape"""

    layer = layer.view(N, -1, C, H, W)

    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]

    layer = layer.reshape(N, -1, C)

    return layer





def concat_box_prediction_layers(box_cls, box_regression):

    # type: (List[Tensor], List[Tensor])

    box_cls_flattened = []

    box_regression_flattened = []



    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):

        # [batch_size, anchors_num_per_position * classes_num, height, width]

        N, AxC, H, W = box_cls_per_level.shape

        Ax4 = box_regression_per_level.shape[1]

        A = Ax4 // 4

        # classes_num

        C = AxC // A



        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)

        box_cls_flattened.append(box_cls_per_level)



        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)

        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)

    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression





class RegionProposalNetwork(nn.Module):

    __annotations__ = {

        'box_coder': BoxCoder,

        'proposal_matcher': Matcher,

        'fg_bg_sampler': BalancedPositiveNegativeSampler,

        'pre_nms_top_n': Dict[str, int],

        'post_nms_top_n': Dict[str, int],

    }



    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh,

                 batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh):

        super(RegionProposalNetwork, self).__init__()

        self.anchor_generator = anchor_generator

        self.head = head

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))



        self.box_similarity = box_iou



        self.proposal_matcher = Matcher(

            fg_iou_thresh,

            bg_iou_thresh,

            allow_low_quality_matches=True

        )



        self.fg_bg_sampler = BalancedPositiveNegativeSampler(

            batch_size_per_image, positive_fraction  # 256, 0.5

        )



        # use during testing

        self._pre_nms_top_n = pre_nms_top_n

        self._post_nms_top_n = post_nms_top_n

        self.nms_thresh = nms_thresh

        self.min_size = 1e-3



    def pre_nms_top_n(self):

        if self.training:

            return self._pre_nms_top_n['training']

        return self._pre_nms_top_n['testing']



    def post_nms_top_n(self):

        if self.training:

            return self._post_nms_top_n['training']

        return self._post_nms_top_n['testing']



    def assign_targets_to_anchors(self, anchors, targets):

        # type: (List[Tensor], List[Dict[str, Tensor]])

        labels = []

        matched_gt_boxes = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):

            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:

                device = anchors_per_image.device

                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)

                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)

            else:

                # Calculate the IOU information of anchors and real bbox

                match_quality_matrix = box_iou(gt_boxes, anchors_per_image)

                matched_idxs = self.proposal_matcher(match_quality_matrix)



                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]



                labels_per_image = matched_idxs >= 0

                labels_per_image = labels_per_image.to(dtype=torch.float32)



                # background (negative examples)

                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD

                labels_per_image[bg_indices] = torch.tensor(0.0)



                # discard indices that are between thresholds

                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2

                labels_per_image[inds_to_discard] = torch.tensor(-1.0)



            labels.append(labels_per_image)

            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes



    def _get_top_n_idx(self, objectness, num_anchors_per_level):

        # type: (Tensor, List[int])

        r = []  

        offset = 0

        for ob in objectness.split(num_anchors_per_level, 1):

            if torchvision._is_tracing():

                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())

            else:

                num_anchors = ob.shape[1]

                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)  # self.pre_nms_top_n=1000

            # Returns the k largest elements of the given input tensor along a given dimension

            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)

            r.append(top_n_idx + offset)

            offset += num_anchors

        return torch.cat(r, dim=1)



    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):

        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int])

        num_images = proposals.shape[0]

        device = proposals.device



        objectness = objectness.detach()

        objectness = objectness.reshape(num_images, -1)



        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)

                  for idx, n in enumerate(num_anchors_per_level)]

        levels = torch.cat(levels, 0)



        # Expand this tensor to the same size as objectness

        levels = levels.reshape(1, -1).expand_as(objectness)



        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)



        image_range = torch.arange(num_images, device=device)

        batch_idx = image_range[:, None]  # [batch_size, 1]



        objectness = objectness[batch_idx, top_n_idx]

        levels = levels[batch_idx, top_n_idx]

        proposals = proposals[batch_idx, top_n_idx]



        final_boxes = []

        final_scores = []

        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):

            boxes = clip_boxes_to_image(boxes, img_shape)

            keep = remove_small_boxes(boxes, self.min_size)

            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level

            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions

            keep = keep[: self.post_nms_top_n()]

            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)

            final_scores.append(scores)

        return final_boxes, final_scores



    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):

        # type: (Tensor, Tensor, List[Tensor], List[Tensor])

        """

        Calculate RPN loss

        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)

        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)



        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()



        labels = torch.cat(labels, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)



        box_loss = smooth_l1_loss(

            pred_bbox_deltas[sampled_pos_inds],

            regression_targets[sampled_pos_inds],

            beta=1 / 9,

            size_average=False,

        ) / (sampled_inds.numel())



        objectness_loss = F.binary_cross_entropy_with_logits(

            objectness[sampled_inds], labels[sampled_inds]

        )



        return objectness_loss, box_loss



    def forward(self, images, features, targets=None):

        # type: (ImageList, Dict[str, Tensor], Optional[List[Dict[str, Tensor]]])

        features = list(features.values())



        objectness, pred_bbox_deltas = self.head(features)



        anchors = self.anchor_generator(images, features)



        # batch_size

        num_images = len(anchors)



        # numel() Returns the total number of elements in the input tensor.

        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]

        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]



        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)



        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)

        proposals = proposals.view(num_images, -1, 4)



        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)



        losses = {}

        if self.training:

            assert targets is not None

            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(

                objectness, pred_bbox_deltas, labels, regression_targets

            )

            losses = {

                "loss_objectness": loss_objectness,

                "loss_rpn_box_reg": loss_rpn_box_reg

            }

        return boxes, losses

import torch

import torch.nn as nn

from torch import Tensor

import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign

from torch.jit.annotations import Tuple, List, Dict, Optional

from collections import OrderedDict

import warnings





class FasterRCNNBase(nn.Module):

    def __init__(self, backbone, rpn, roi_heads, transform):

        super(FasterRCNNBase, self).__init__()

        self.backbone = backbone

        self.rpn = rpn

        self.roi_heads = roi_heads

        self.transform = transform

        self._has_warned = False



    @torch.jit.unused

    def eager_outputs(self, losses, detections):

        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]

        if self.training:

            return losses



        return detections



    def forward(self, images, targets=None):

        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        if self.training and targets is None:

            raise ValueError("In training mode, targets should be passed")



        if self.training:

            assert targets is not None

            for target in targets:

                boxes = target["boxes"]

                if isinstance(boxes, torch.Tensor):

                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:

                        raise ValueError("Expected target boxes to be a tensor of shape [N, 4], got {:}.".format(boxes.shape))

                else:

                    raise ValueError("Expected target boxes to be of type Tensor, got {:}.".format(type(boxes)))



        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])

        for img in images:

            val = img.shape[-2:]

            assert len(val) == 2

            original_image_sizes.append((val[0], val[1]))



        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):

            features = OrderedDict([('0', features)])



        proposals, proposal_losses = self.rpn(images, features, targets)



        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)



        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)



        losses = {}

        losses.update(detector_losses)

        losses.update(proposal_losses)



        if torch.jit.is_scripting():

            if not self._has_warned:

                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")

                self._has_warned = True

            return losses, detections

        else:

            return self.eager_outputs(losses, detections)





class TwoMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size):

        super(TwoMLPHead, self).__init__()



        self.fc6 = nn.Linear(in_channels, representation_size)

        self.fc7 = nn.Linear(representation_size, representation_size)



    def forward(self, x):

        x = x.flatten(start_dim=1)



        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))



        return x





class FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):

        super(FastRCNNPredictor, self).__init__()



        self.cls_score = nn.Linear(in_channels, num_classes)

        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)



    def forward(self, x):

        if x.dim() == 4:

            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)

        scores = self.cls_score(x)

        bbox_deltas = self.bbox_pred(x)



        return scores, bbox_deltas





class FasterRCNN(FasterRCNNBase):

    def __init__(self, backbone, num_classes=None,

                 # transform parameter

                 min_size=800, max_size=1344,

                 image_mean=None, image_std=None,

                 # RPN parameters

                 rpn_anchor_generator=None, rpn_head=None,

                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,

                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,

                 rpn_nms_thresh=0.7,

                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,

                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

                 # Box parameters

                 box_roi_pool=None, box_head=None, box_predictor=None,

                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,

                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,

                 box_batch_size_per_image=512, box_positive_fraction=0.25,

                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):

            raise ValueError(

                "backbone should contain an attribute out_channels"

                "specifying the number of output channels  (assumed to be the"

                "same for all the levels"

            )



        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))

        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))



        if num_classes is not None:

            if box_predictor is not None:

                raise ValueError("num_classes should be None when box_predictor is specified")

        else:

            if box_predictor is None:

                raise ValueError("num_classes should not be None when box_predictor is not specified")



        out_channels = backbone.out_channels



        if rpn_anchor_generator is None:

            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

            rpn_anchor_generator = AnchorsGenerator(

                anchor_sizes, aspect_ratios

            )



        if rpn_head is None:

            rpn_head = RPNHead(

                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]

            )



        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)

        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)



        rpn = RegionProposalNetwork(

            rpn_anchor_generator, rpn_head,

            rpn_fg_iou_thresh, rpn_bg_iou_thresh,

            rpn_batch_size_per_image, rpn_positive_fraction,

            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)



        if box_roi_pool is None:

            box_roi_pool = MultiScaleRoIAlign(

                featmap_names=['0', '1', '2', '3'],

                output_size=[7, 7],

                sampling_ratio=2)



        if box_head is None:

            resolution = box_roi_pool.output_size[0]

            representation_size = 1024

            box_head = TwoMLPHead(

                out_channels * resolution ** 2,

                representation_size

            )



        if box_predictor is None:

            representation_size = 1024

            box_predictor = FastRCNNPredictor(

                representation_size,

                num_classes)



        roi_heads = ROIHeads(

            # box

            box_roi_pool, box_head, box_predictor,

            box_fg_iou_thresh, box_bg_iou_thresh,

            box_batch_size_per_image, box_positive_fraction,

            bbox_reg_weights,

            box_score_thresh, box_nms_thresh, box_detections_per_img)



        if image_mean is None:

            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:

            image_std = [0.229, 0.224, 0.225]



        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)



        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
def create_model(num_classes):

    backbone = resnet50_fpn_backbone()

    model = FasterRCNN(backbone=backbone, num_classes=91)

    

    weights_dict = torch.load("../input/weights/5-resNetFpn-model-19.pth")

    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

    if len(missing_keys) != 0 or len(unexpected_keys) != 0:

        print("missing_keys: ", missing_keys)

        print("unexpected_keys: ", unexpected_keys)



    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model
import torch

import os

import time

import warnings



warnings.filterwarnings('ignore')





def collate_fn(batch):

    return tuple(zip(*batch))





def main():

    since = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    print(device)



    data_transform = {

        "train": transforms.Compose([transforms.ToTensor(),

                                     transforms.RandomHorizontalFlip(0.5)]),

        "val": transforms.Compose([transforms.ToTensor()])

    }



    data_root = "../input/global-wheat-detection"

    # load train data set

    train_data_set = WheatDataSet(data_root, data_transform["train"])

    train_data_loader = torch.utils.data.DataLoader(train_data_set,

                                                    batch_size=4,

                                                    shuffle=True,

                                                    num_workers=2,

                                                    collate_fn=collate_fn)



    # create model num_classes equal background + 20 classes

    model = create_model(num_classes=2)



    model.to(device)



    # define optimizer

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                                   step_size=5,

                                                   gamma=0.33)

    # You can install pycocotools to simplify your training.I omitted these steps
import torch

import torchvision

from torchvision import transforms

from PIL import Image

import json

import matplotlib.pyplot as plt

import pandas as pd

import time

import warnings

import glob



warnings.filterwarnings('ignore')





def create_model(num_classes):

    backbone = resnet50_fpn_backbone()

    model = FasterRCNN(backbone=backbone, num_classes=num_classes)



    return model





# get devices

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



# create model

model = create_model(num_classes=2)



# load train weights

train_weights = "../input/weights/15-fasterrcnn_resnet50_fpn_best.pth"

# model.load_state_dict(torch.load(train_weights)["model"])

model.load_state_dict(torch.load(train_weights))

model.to(device)



# read class_indict

category_index = {"wheat": 1}





origin_list = glob.glob('../input/global-wheat-detection/test/*.jpg')

print(origin_list)



preds = []



for i, img_name in enumerate(origin_list):

    # load image

    original_img = Image.open(img_name)



    # from pil image to tensor, do not normalize image

    data_transform = transforms.Compose([transforms.ToTensor()])

    img = data_transform(original_img)

    # expand batch dimension

    img = torch.unsqueeze(img, dim=0)



    model.eval()

    with torch.no_grad():

        since = time.time()

        predictions = model(img.to(device))[0]

        print('{} Time:{}s'.format(i, time.time() - since))

        predict_boxes = predictions["boxes"].to("cpu").numpy()

        predict_classes = predictions["labels"].to("cpu").numpy()

        predict_scores = predictions["scores"].to("cpu").numpy()



        draw_box(original_img,

                 predict_boxes,

                 predict_classes,

                 predict_scores,

                 category_index,

                 thresh=0.5,

                 line_thickness=5)

        plt.imshow(original_img)

        plt.show()



        predict = ""

        for box, score in zip(predict_boxes, predict_scores):

            str_box = ""

            box[2] = box[2] - box[0]

            box[3] = box[3] - box[1]

            for b in box:

                str_box += str(b) + ' '

            predict += str(score) + ' ' + str_box

        preds.append(predict)



name_list = [name.split('/')[-1].split('.')[0] for name in origin_list]

print(name_list)

dataframe = pd.DataFrame({"image_id": name_list, "PredictionString": preds})

dataframe.to_csv('submission.csv', index=False)