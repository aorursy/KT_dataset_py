import pandas as pd

import numpy as np

import cv2

import os

import torch

from torchvision import models



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt
!git clone https://github.com/cocodataset/cocoapi.git

%cd /kaggle/working/cocoapi/PythonAPI

!python setup.py build_ext install
# %%shell

%cd /kaggle/working/



!pip install cython

# Install pycocotools, the version by default in Colab

# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Download TorchVision repo to use some files from

# references/detection

!git clone https://github.com/pytorch/vision.git

%cd vision

!git checkout v0.3.0



!cp references/detection/utils.py ../

!cp references/detection/transforms.py ../

!cp references/detection/coco_eval.py ../

!cp references/detection/engine.py ../

!cp references/detection/coco_utils.py ../
from engine import train_one_epoch, evaluate

import utils

import transforms as T
train_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

print(train_df.shape)

# test_df = pd.read_csv('/kaggle/input/global-wheat-detection/test.csv')

# print(test_df.shape)
img_ids = train_df['image_id'].unique()

valid_ids = img_ids[-300:]

train_ids = img_ids[:-300]

valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]

valid_df.shape,train_df.shape
## A TEST TO IMROVE THE SCORE - NORMALIZATION OF THE IMAGE PIXEL VALUES (DIVISION BY 255)



# img = cv2.imread("/kaggle/input/global-wheat-detection/test/2fd875eaa.jpg")

# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# plt.figure()

# plt.imshow(img)

# i = img/255

# plt.figure()

# plt.imshow(i)

# print(img,i)
class GWDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.img_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index):

        img_idx = self.img_ids[index]

        img_name = str(img_idx+'.jpg')

        # load images ad masks

        img_path = os.path.join(self.image_dir, img_name)

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)

        img = img/255.0

       # get bounding box coordinates for each mask

        num_bbxs = len(self.df[self.df['image_id']==img_idx])

        bbxs = self.df[self.df['image_id']==img_idx]

        boxes = []

        area = []

#         print(bbxs)

        for t in range(num_bbxs):

            l = bbxs.iloc[t]['bbox'].split(',')

#             print(l)

            xmin,ymin,w,h = float(l[0][1:]),float(l[1][1:]),float(l[2][1:]),float(l[3][1:-1])

            xmax = xmin+w

            ymax = ymin+h

            area.append(w*h)

            boxes.append([xmin, ymin, xmax, ymax])



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class

        labels = torch.ones((num_bbxs,), dtype=torch.int64)



        imag_id = torch.tensor([index])

        # suppose all instances are not crowd

        area = torch.as_tensor(area, dtype=torch.float32)

        iscrowd = torch.zeros((num_bbxs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["image_id"] = imag_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target



    def __len__(self):

        return (self.img_ids.shape[0])
def get_transform(train):

    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor

    transforms.append(T.ToTensor())

    if train:

        # during training, randomly flip the training images

        # and ground-truth for data augmentation

        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
dataset = GWDataset(valid_df,'/kaggle/input/global-wheat-detection/train/',get_transform(train = False))

dataset[0]
##  CHOICE BETWEEN MODEL PRETRAINED ON image_net vs Global Wheat Detection Challenge

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model = torch.load("/kaggle/input/gwd-model/fasterrcnn_resnet50_fpn.pth",map_location='cpu')
num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# get_train_transform()=

def collate_fn(batch):

    return tuple(zip(*batch))

DIR_TRAIN = '/kaggle/input/global-wheat-detection/train/'

train_dataset = GWDataset(train_df, DIR_TRAIN,get_transform(train = True))

valid_dataset = GWDataset(valid_df, DIR_TRAIN, get_transform(train = False))





# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=8,

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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

num_epochs = 2

num_epochs = 4



for epoch in range(num_epochs):

    # train for one epoch, printing every 10 iterations

    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)

    # update the learning rate

    lr_scheduler.step()

    # evaluate on the test dataset

    evaluate(model, valid_data_loader, device=device)
# pick one image from the test set

img, _ = valid_dataset[0]

# put the model in evaluation mode

model.eval()

with torch.no_grad():

    prediction = model([img.to(device)])

sample = valid_dataset[0][0].permute(1,2,0).numpy()

boxes = prediction[0]['boxes'].cpu().numpy().astype(np.int32)

# boxe = boxes.reshape((4,-1))

scores = prediction[0]['scores'].cpu().numpy()

boxes.shape
scores
# plt.imshow(sample)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))



color = (220,0,0)

for i in range(len(boxes)):

#     print(type(box[0]))

    if scores[i]>0.97:

        cv2.rectangle(img,(int(boxes[i][0]), int(boxes[i][1])),(int(boxes[i][2]), int(boxes[i][3])),color, 5)

ax.set_axis_off()

ax.imshow(img)
torch.save(model, '/kaggle/working/fasterrcnn_resnet50_fpn_new.pth')

torch.save(model.state_dict(), '/kaggle/working/fasterrcnn_resnet50_fpn_statedict.pth')