import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import cv2

from PIL import Image

import torch

import torch.utils.data

from torchvision.transforms import functional as F

import torchvision.transforms as transforms

import re

from torch.utils.data import DataLoader, Dataset
DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



#  将bbox的四个数转成x、y、w、h

def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



#  将bbox的四个数转成x、y、w、h

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

# 丢弃bbox这一栏，换成用x、y、w、h

train_df.drop(columns=['bbox'], inplace=True)

# 边框转成float形式

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
image_ids = train_df['image_id'].unique() #去重

# 划分数据集，先用id来划分

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]
# 再根据id来划分

valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
# 继承了torch的Dataset

class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()

        self.image_ids = dataframe['image_id'].unique() # 图像ID

        self.df = dataframe #数据集

        self.image_dir = image_dir

        self.transforms = transforms #图像转化工具



    # 通过下标获取一张图片的信息

    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]

        

        # 读图片，并做一些转化

        image = Image.open(f'{self.image_dir}/{image_id}.jpg').convert("RGB")

        image = np.array(image)

        image = F.to_tensor(image)



        # 处理一下边框，转换成左下和右上两个点

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes = torch.as_tensor(boxes, dtype=torch.float32)



        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        # 边框的面积

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        # 所有标签都为1

        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # 这个是什么？

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        # 将这张图片的信息放在target字典里面

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        return image, target





    def __len__(self) -> int:

        return self.image_ids.shape[0]
# 这个函数要来干嘛？

def collate_fn(batch):

    return tuple(zip(*batch))



# 训练集和验证集

train_dataset = WheatDataset(train_df, DIR_TRAIN)

valid_dataset = WheatDataset(valid_df, DIR_TRAIN)



# 从训练集中再分割出一部分作为测试集 

# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



# our dataset has two classes only - background and person

num_classes = 2

# get the model using our helper function



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





# move model to the right device

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)



# 大疑问：损失函数是什么？

# construct an optimizer

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)



# and a learning rate scheduler which decreases the learning rate by

# 10x every 3 epochs

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
num_epochs = 20



for epoch in range(num_epochs): 

    sum_loss = 0.0

    cal_freq = 10

    itr = 1

    for images, targets in train_data_loader: # 我猜就是一下子给batch_size个数据，迭代size/batch_size次

        # 为什么要做这两步工作？ 为了.to(device)，转移到cuda:0上

        # 获取图像本体和信息

        images = list(image.to(device) for image in images)

        # 是一个list，每个元素是一个dict，对应一张image

        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]



        # 计算损失值

        # 疑问1：损失函数是什么？在哪里设定？

        # 疑问2：输入图像的大小？在哪里设定？

        loss_dict = model(images, targets)



        # 统计损失值之和，并梯度下降

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad() # ？？？

        losses.backward()  #反向传播

        optimizer.step() # ？？？



        # 用于输出训练过程 

        sum_loss += losses.item() # 损失值累加

        if itr % cal_freq == 0:

            print("Iteration %d loss:%.03f"%(itr,sum_loss/cal_freq))

            sum_loss = 0

        itr += 1



    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()

    print('--------------Epoch %d finished'%epoch) 
images, targets = next(iter(valid_data_loader))

images = list(image.to(device) for image in images)  #转到GPU上

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  #转到GPU上
test_index = 0

boxes = targets[test_index]['boxes'].cpu().numpy().astype(np.int32)

#  这个permute函数用来干什么？ 哦，原本通道是放到第一维的，现在放到第三维？

in_sample = images[test_index]

sample = in_sample.permute(1,2,0).cpu().numpy()
# put the model in evaluation mode

model.eval()

with torch.no_grad():

    prediction = model([in_sample.to(device)])[0]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in prediction['boxes']:

        cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 3)

ax.set_axis_off()

ax.imshow(sample)