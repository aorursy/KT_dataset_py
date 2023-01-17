%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

from torchvision import models

from torch import nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torch.nn.functional as F



from PIL import Image, ImageDraw

import matplotlib.pyplot as plt



import os

import json

import re

import random

from math import sqrt



import os

print(os.listdir("../input"))

with open('../input/annotations/Annotation/n02113799-standard_poodle/n02113799_489') as f:

    reader = f.read()
img = Image.open('../input/images/Images/n02113799-standard_poodle/n02113799_489.jpg')



xmin = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)[0])

xmax = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)[0])

ymin = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)[0])

ymax = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)[0])



origin_img = img.copy()

draw = ImageDraw.Draw(origin_img)

draw.rectangle(xy=[(xmin,ymin), (xmax,ymax)])

origin_img
all_img_folder = os.listdir('../input/images/Images')

# all_annotation_folder = os.listdir('../input/annotations/Annotation')



all_img_name = []

for img_folder in all_img_folder:

    img_folder_path = '../input/images/Images/' + img_folder

    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))



# all_annotation_name = []

# for annotation_folder in all_annotation_folder:

#     annotation_folder_path = '../input/annotations/Annotation/' + annotation_folder

#     all_annotation_name += list(map(lambda x: annotation_folder + '/'+ x, os.listdir(annotation_folder_path)))



len(all_img_name), all_img_name[0]
class MyDateset(Dataset):

    def __init__(self, file_folder, is_test=False, transform=None):

        self.img_folder_path = '../input/images/Images/'

        self.annotation_folder_path = '../input/annotations/Annotation/'

        self.file_folder = file_folder

        self.transform = transform

        self.is_test = is_test

        

    def __getitem__(self, idx):

        file = self.file_folder[idx]

        img_path = self.img_folder_path + file

        img = Image.open(img_path).convert('RGB')

        

        if not self.is_test:

            annotation_path = self.annotation_folder_path + file.split('.')[0]

            with open(annotation_path) as f:

                annotation = f.read()



            xy = self.get_xy(annotation)

            box = torch.FloatTensor(list(xy))



            new_box = self.box_resize(box, img)

            if self.transform is not None:

                img = self.transform(img)



            return img, new_box

        else:

            if self.transform is not None:

                img = self.transform(img)

            return img

    

    def __len__(self):

        return len(self.file_folder)

        

    def get_xy(self, annotation):

        xmin = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', annotation)[0])

        xmax = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', annotation)[0])

        ymin = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', annotation)[0])

        ymax = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', annotation)[0])

        

        return xmin, ymin, xmax, ymax

    

    def show_box(self):

        file = random.choice(self.file_folder)

        annotation_path = self.annotation_folder_path + file.split('.')[0]

        

        img_box = Image.open(self.img_folder_path + file)

        with open(annotation_path) as f:

            annotation = f.read()

            

        draw = ImageDraw.Draw(img_box)

        xy = self.get_xy(annotation)

        print('bbox:', xy)

        draw.rectangle(xy=[xy[:2], xy[2:]])

        

        return img_box

        

    def box_resize(self, box, img, dims=(332, 332)):

        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)

        new_box = box / old_dims

        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)

        new_box = new_box * new_dims

        

        return new_box
tsfm = transforms.Compose([

    transforms.Resize([332, 332]),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])



test_tsfm = transforms.Compose([

    transforms.Resize([332, 332]),

    transforms.ToTensor(),

#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])
class Mymodel(nn.Module):

    def __init__(self):

        super().__init__()

        self.body = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

        self.head = nn.Sequential(

            nn.BatchNorm1d(2048*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Dropout(0.25),

            nn.Linear(in_features=2048*2, out_features=1024, bias=True),

            nn.ReLU(),

            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Dropout(0.5),

            nn.Linear(in_features=1024, out_features=4, bias=True),

        )

        

    def forward(self, x):

        x = self.body(x)

        x1 = nn.AdaptiveAvgPool2d(1)(x)

        x2 = nn.AdaptiveMaxPool2d(1)(x)

        x = torch.cat([x1,x2], 1)

        x = x.view(x.size(0), -1)

        x = self.head(x)

        

        return x
def loss_fn(preds, targs, class_idxs):

    return nn.L1Loss()(preds, targs.squeeze())



def IoU(preds, targs):

    return intersection(preds, targs.squeeze()) / union(preds, targs.squeeze())



def intersection(preds, targs):

    # preds and targs are of shape (bs, 4), pascal_voc format

    if len(targs.shape) == 1:

        targs = targs.reshape(1, 4)

#     print(preds.shape, targs.shape)

    max_xy = torch.min(preds[:, 2:], targs[:, 2:])

    min_xy = torch.max(preds[:, :2], targs[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, 0] * inter[:, 1]



def area(boxes):

    return ((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))



def union(preds, targs):

    if len(targs.shape) == 1:

        targs = targs.reshape(1, 4)

    return area(preds) + area(targs) - intersection(preds, targs)
EPOCH = 10

BS = 64

LR = 1e-1
train_ds = MyDateset(all_img_name[:int(len(all_img_name)*0.8)], transform=tsfm)

train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)



valid_ds = MyDateset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)], transform=tsfm)

valid_dl = DataLoader(valid_ds, batch_size=1)



test_ds = MyDateset(all_img_name[int(len(all_img_name)*0.9):], is_test=True, transform=test_tsfm)

test_dl = DataLoader(test_ds, batch_size=1)
train_ds.show_box()
model = Mymodel()

model.cuda();



for layer in model.body.parameters():

    layer.requires_grad = False

    

optm = torch.optim.Adam(model.head.parameters(), lr=LR)
from tqdm import tqdm

model.train()

for epoch in range(1, EPOCH+1):

    if epoch == 3:

        optm = torch.optim.Adam(model.head.parameters(), lr=LR/2)

    elif  epoch == 5:

        optm = torch.optim.Adam(model.head.parameters(), lr=LR/4)

    elif epoch == 7:

        optm = torch.optim.Adam(model.head.parameters(), lr=LR/8)

    elif epoch == 9:

        optm = torch.optim.Adam(model.head.parameters(), lr=LR/10)

        

    train_loss = []

    train_IoU = []

    for step, data in enumerate(train_dl):

        imgs, boxes = data

        imgs = imgs.cuda()

        boxes = boxes.cuda()

        

        pred = model(imgs)

        loss = nn.L1Loss()(pred, boxes.squeeze())

        train_loss.append(loss.item())

        IOU = IoU(pred, boxes)

        train_IoU.append(IOU)

            

        optm.zero_grad()

        loss.backward()

        optm.step()

        if step % 10 == 0:

            print('step: ', step, '/', len(train_dl), '\tloss:', loss.item(), '\tIoU:', float(IOU.mean()))

        

    model.eval()

    valid_loss = []

    valid_IoU = []

    for step, data in enumerate(tqdm(valid_dl)):

        imgs, boxes = data

        imgs = imgs.cuda()

        boxes = boxes.cuda()

        

        pred = model(imgs)

        loss = nn.L1Loss()(pred, boxes.squeeze())

        valid_loss.append(loss.item())

        IOU = IoU(pred, boxes)

        valid_IoU.append(IOU.item())

    print('epoch:', epoch, '/', EPOCH, '\ttrain_loss:', np.mean(train_loss), '\tvalid_loss:', np.mean(valid_loss), '\tIoU:', np.mean(valid_IoU))
model.eval()



draw_img = []

for step, img in enumerate(tqdm(test_dl)):

    img = img.cuda()

    pred = model(img)

    

    origin_img = transforms.ToPILImage()(img.cpu().squeeze())

    draw = ImageDraw.Draw(origin_img)

    xmin, ymin, xmax, ymax = tuple(pred.squeeze().tolist())

    draw.rectangle(xy=[(int(xmin),int(ymin)), (int(xmax),int(ymax))])

    draw_img.append(origin_img)
draw_img[14]