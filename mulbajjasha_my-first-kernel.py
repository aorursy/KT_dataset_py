import json

import pandas as pd
with open('/kaggle/input/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file: ## using with to remove works of json

    label_desc = json.load(file)

sample_sub_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/sample_submission.csv')

train_csv_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv')
label_desc
label_desc.keys()
num_classes = len(label_desc['categories'])

num_attributes = len(label_desc['attributes'])

print(f'Total # of classes: {num_classes}')

print(f'Total # of attributes: {num_attributes}')
categories_df = pd.DataFrame(label_desc.get('categories'))

attributes_df = pd.DataFrame(label_desc['attributes'])

categories_df
attributes_df
pd.set_option('display.max_rows' , 294)

attributes_df
import seaborn as sns

import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize = (20,20))

sns.countplot('supercategory',data = attributes_df,ax = ax[0])

sns.countplot('supercategory',data = categories_df,ax = ax[1])

plt.show()
fig,ax = plt.subplots(1,2,figsize = (20,20))

sns.countplot('level',data = attributes_df,ax = ax[0])

sns.countplot('level',data = categories_df,ax = ax[1])

plt.show()
sample_sub_df
train_csv_df
train_csv = train_csv_df.groupby('ImageId')['Height','Width'].first().hist(bins = 100)

train_csv
pd.DataFrame([train_csv_df['Height'].describe(), train_csv_df['Width'].describe()]).T.loc[['min','max','mean']]
import torch

import torch.utils
import collections

from tqdm import tqdm
def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array

    shape: (height,width) of array to return

    Returns numpy array according to the shape, 1 - mask, 0 - background

    '''

    shape = (shape[1], shape[0])

    s = mask_rle.split()

    # gets starts & lengths 1d arrays

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    starts -= 1

    # gets ends 1d array

    ends = starts + lengths

    # creates blank mask image 1d array

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # sets mark pixles

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    # reshape as a 2d mask image

    return img.reshape(shape).T  # Needed to align to RLE direction







class FashionDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df_path, height, width, transforms=None):

        self.transforms = transforms

        self.image_dir = image_dir

        self.df = pd.read_csv(df_path, nrows=10000)

        self.height = height

        self.width = width

        self.image_info = collections.defaultdict(dict)

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])

        temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()

        size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()

        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):

            image_id = row['ImageId']

            image_path = os.path.join(self.image_dir, image_id)

            self.image_info[index]["image_id"] = image_id

            self.image_info[index]["image_path"] = image_path

            self.image_info[index]["width"] = self.width

            self.image_info[index]["height"] = self.height

            self.image_info[index]["labels"] = row["CategoryId"]

            self.image_info[index]["orig_height"] = row["Height"]

            self.image_info[index]["orig_width"] = row["Width"]

            self.image_info[index]["annotations"] = row["EncodedPixels"]



    def __getitem__(self, idx):

        # load images ad masks

        img_path = self.image_info[idx]["image_path"]

        img = Image.open(img_path).convert("RGB")

        img = img.resize((self.width, self.height), resample=Image.BILINEAR)



        info = self.image_info[idx]

        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)

        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):

            sub_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))

            sub_mask = Image.fromarray(sub_mask)

            sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BILINEAR)

            mask[m, :, :] = sub_mask

            labels.append(int(label) + 1)



        num_objs = len(labels)

        boxes = []

        new_labels = []

        new_masks = []



        for i in range(num_objs):

            try:

                pos = np.where(mask[i, :, :])

                xmin = np.min(pos[1])

                xmax = np.max(pos[1])

                ymin = np.min(pos[0])

                ymax = np.max(pos[0])

                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:

                    boxes.append([xmin, ymin, xmax, ymax])

                    new_labels.append(labels[i])

                    new_masks.append(mask[i, :, :])

            except ValueError:

                continue



        if len(new_labels) == 0:

            boxes.append([0, 0, 20, 20])

            new_labels.append(0)

            new_masks.append(mask[0, :, :])



        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)

        for i, n in enumerate(new_masks):

            nmx[i, :, :] = n



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(new_labels, dtype=torch.int64)

        masks = torch.as_tensor(nmx, dtype=torch.uint8)



        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target



    def __len__(self):

        return len(self.image_info)
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(),

                                transforms.RandomApply([transforms.Resize([1333,1333]),

                                                      transforms.RandomHorizontalFlip(),

                                                      transforms.ColorJitter(brightness= 0.3, contrast= 0.3,saturation=0.1, hue=0.1)],

                                                       p=0.5),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                      ])
num_classes = 46 + 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
dataset_train = FashionDataset("/kaggle/input/imaterialist-fashion-2020-fgvc7/train/",

                               "/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv",

                               512,

                               512,

                               transforms=transform

                              )
from PIL import Image, ImageFile

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
model_ft =torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

in_features = model_ft.roi_heads.box_predictor.cls_score.in_features

model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256

model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

model_ft.to(device)

for param in model_ft.parameters():

    param.requires_grad = True
data_loader = torch.utils.data.DataLoader(

    dataset_train, batch_size=16, shuffle=True, num_workers=8,

    collate_fn=lambda x: tuple(zip(*x)))
!pip install -U pytorch_warmup

import pytorch_warmup as warmup
for param in model_ft.parameters():

    param.requires_grad = False

model_ft.eval()

optimizer = torch.optim.SGD(params, lr=0.03, momentum=0.9, weight_decay=0.0001)
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):

    model.train()

    metric_logger = MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)



    lr_scheduler = None

    if iterations < 500: 

        lr = warmup(warmup_factor = 1. / 3) 

    if epochs == 10: lr = warmup(warmup_factor = 1. / 10) 

    if epochs == 18: lr = warmup(warmup_factor = 1. / 10) 

    if epochs > 20: stop

    lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, step_size=5, gamma=0.1)



    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())



        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        if lr_scheduler is not None:

            lr_scheduler.step()



        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
from __future__ import print_function



from collections import defaultdict, deque

import datetime

import pickle

import time

import torch.distributed as dist

import errno



import collections

import os

import numpy as np

import torch

import torch.utils.data

from PIL import Image, ImageFile

import pandas as pd

from tqdm import tqdm

from torchvision import transforms

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
num_epochs = 20

for epoch in range(num_epochs):

    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=100)

    lr_scheduler.step()