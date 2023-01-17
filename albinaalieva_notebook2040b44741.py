import numpy as np 

import pandas as pd 

import os

import time

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import albumentations as alb

from albumentations.pytorch.transforms import ToTensorV2

import torch

from torch.utils.data import DataLoader,Dataset

from torch.utils.data import SubsetRandomSampler

import torchvision 

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection import FasterRCNN

import cv2

from tqdm.notebook import tqdm

import torch.nn as nn

train_path = '/kaggle/input/global-wheat-detection/train.csv'

train_img_path = '/kaggle/input/global-wheat-detection/train'
train = pd.read_csv(train_path)

train.head()
train['image_id'] = train['image_id'].apply(lambda x: str(x) + '.jpg')
bboxes = np.stack(train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep = ',')))

for i, col in enumerate(['x_min', 'y_min', 'w', 'h']):

    train[col] = bboxes[:,i]



train.drop(columns = ['bbox'], inplace = True)

train.head()
train['bbox_area'] = train['w']*train['h']
max_area = 100000

min_area = 40

train_clean = train[(train['bbox_area'] < max_area) & (train['bbox_area'] > min_area)]
train_split = 0.8



image_ids = train_clean['image_id'].unique()

train_ids = image_ids[0:int(train_split*len(image_ids))]

val_ids = image_ids[int(train_split*len(image_ids)):]



print('Length of training ids', len(train_ids))

print('Length of validation ids', len(val_ids))
train_df = train_clean[train_clean['image_id'].isin(train_ids)]

valid_df = train_clean[train_clean['image_id'].isin(val_ids)]
class WheatDataset(Dataset):

    def __init__(self, df, image_dir,transform = None):

        super().__init__()

        self.df = df

        self.img_ids = df['image_id'].unique()

        self.image_dir = image_dir

        self.transform = transform

        

    def __len__(self):

        return len(self.img_ids)

    

    def __getitem__(self, idx: int):

        image_id = self.img_ids[idx]

        pts = self.df[self.df['image_id'] == image_id]

        

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image = image/255.0

        

        boxes = pts[['x_min', 'y_min', 'w', 'h']].values

        

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 

        area = torch.as_tensor(area, dtype = torch.float32)

        

        labels = torch.ones((pts.shape[0],), dtype=torch.int64)

        

        iscrowd = torch.zeros((pts.shape[0],), dtype=torch.int32)

        

        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["image_id"] = torch.tensor(idx)

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transform:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': target['labels']

            }

            sample = self.transform(**sample)

            image = sample['image']

            

            if len(sample['bboxes']) > 0:

                target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)

            else:

                target['boxes'] = torch.linspace(0,3, steps = 4, dtype = torch.float32)

                target['boxes'] = target['boxes'].reshape(-1,4)

            

        return image, target, image_id
def get_training_transforms():

    return alb.Compose([

    alb.VerticalFlip(p = 0.5),

    alb.HorizontalFlip(p = 0.5),

    ToTensorV2(p = 1.0)

], p=1.0, bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['labels']))



def get_validation_transforms():

    return alb.Compose([ToTensorV2(p = 1.0)], p = 1.0, bbox_params = alb.BboxParams(format='pascal_voc', label_fields=['labels']))
densenet_net = torchvision.models.densenet169(pretrained=True)



modules = list(densenet_net.children())[:-1]

backbone = nn.Sequential(*modules)

backbone.out_channels = 1664



anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),

                                   aspect_ratios=((0.5, 1.0, 2.0),))



roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],

                                                output_size=7,

                                                sampling_ratio=2)

model = FasterRCNN(backbone,

                   num_classes=2,

                   rpn_anchor_generator=anchor_generator,

                   box_roi_pool=roi_pooler)

def collate_fn(batch):

    return tuple(zip(*batch))
training_dataset = WheatDataset(train_df, train_img_path, get_training_transforms())

validation_dataset = WheatDataset(valid_df, train_img_path, get_validation_transforms())



train_dataloader = DataLoader(

        training_dataset, batch_size=2, shuffle= True, num_workers=4,

        collate_fn= collate_fn)



valid_dataloader = DataLoader(

        validation_dataset, batch_size=2, shuffle=False, num_workers=4,

        collate_fn=collate_fn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
images, targets, image_ids = next(iter(train_dataloader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]



optimizer = torch.optim.SGD(params, lr= 0.01, momentum=0.93)



lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, eps=1e-08)

num_epochs=15
total_train_loss = []

total_test_loss = []



for epoch in range(num_epochs):

    model.train()



    print('Epoch: ', epoch + 1)

    train_loss = []

    

    for images, targets, image_ids in tqdm(train_dataloader):

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        

        loss_dict = model(images, targets)  

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        train_loss.append(loss_value)

        

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        

    epoch_loss = np.mean(train_loss)

    print('Epoch Loss is: ' , epoch_loss)

    total_train_loss.append(epoch_loss)

    

    with torch.no_grad():

        test_losses = []

        for images, targets, image_ids in tqdm(valid_dataloader):

            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            

            loss_dict = model(images, targets)



            losses = sum(loss for loss in loss_dict.values())

            test_loss = losses.item()

            test_losses.append(test_loss)

            

    test_losses_epoch = np.mean(test_losses)

    print('Test Loss: ' ,test_losses_epoch)

    total_test_loss.append(test_losses_epoch)

    

    if lr_scheduler is not None:

        lr_scheduler.step(test_losses_epoch)

        

torch.save(model.state_dict(), 'fasterrcnn.pth')
model.eval()