from __future__ import print_function



from collections import defaultdict, deque

import datetime

import pickle

import time

import torch.distributed as dist

import errno

from fastai import metrics



import cv2

import collections

import os

import numpy as np



import torch.utils.data

from PIL import Image, ImageFile

import pandas as pd

from tqdm import tqdm_notebook as tqdm

from torchvision import transforms

import torchvision

import random

from torch.utils.data import DataLoader, Dataset, sampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2

import pdb

import time

import warnings

import random

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

from torch.nn import functional as F

import torch.optim as optim

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset, sampler

from matplotlib import pyplot as plt

from albumentations import (HorizontalFlip,VerticalFlip,Cutout,SmallestMaxSize,

                            ToGray, ShiftScaleRotate, Blur,Normalize, Resize, Compose, GaussNoise)

from albumentations.pytorch import ToTensor



import matplotlib.pyplot as plt

import pandas as pd

import os

from tqdm import tqdm_notebook

import cv2

from PIL import Image



from torchvision import models

from torch.utils.data import DataLoader, Dataset

import torch.utils.data as utils



import platform

print(f'Python version: {platform.python_version()}')

print(f'PyTorch version: {torch.__version__}')
def seed_everything(seed=43):

    '''

      Make PyTorch deterministic.

    '''    

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.deterministic = True



seed_everything()



IS_DEBUG = False
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):



    def f(x):

        if x >= warmup_iters:

            return 1

        alpha = float(x) / warmup_iters

        return warmup_factor * (1 - alpha) + alpha



    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
class DiceLoss(torch.nn.Module):

    def __init__(self):

        super(DiceLoss, self).__init__()

 

    def forward(self, logits, targets):

        ''' fastai.metrics.dice uses argmax() which is not differentiable, so it 

          can NOT be used in training, however it can be used in prediction.

          see https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53

        '''

        N = targets.size(0)

        preds = torch.sigmoid(logits)

        #preds = logits.argmax(dim=1) # do NOT use argmax in training, because it is NOT differentiable

        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/backend.py#L96

        EPSILON = 1e-7

 

        preds_flat = preds.view(N, -1)

        targets_flat = targets.view(N, -1)

 

        intersection = (preds_flat * targets_flat).sum()#.float()

        union = (preds_flat + targets_flat).sum()#.float()

        

        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)

        loss = 1 - loss / N

        return loss




def train_one_epoch(model, optimizer, data_loader, device, epoch):

    model.train()

    loss_func = DiceLoss() #nn.BCEWithLogitsLoss() 



    lr_scheduler = None

    if epoch == 0:

        warmup_factor = 1. / 1000

        warmup_iters = min(1000, len(data_loader) - 1)



        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)



    lossf=None

    inner_tq = tqdm(data_loader, total=len(data_loader), leave=False, desc= f'Iteration {epoch}')

    for images, masks in inner_tq:

        y_preds = model(images.to(device))

        y_preds = y_preds['out'][:, 1, :, :] #



        loss = loss_func(y_preds, masks.to(device))



        if torch.cuda.device_count() > 1:

            loss = loss.mean() # mean() to average on multi-gpu.



        loss.backward()

        optimizer.step()

        optimizer.zero_grad()



        if lr_scheduler is not None:

            lr_scheduler.step()



        if lossf:

            lossf = 0.98*lossf+0.02*loss.item()

        else:

            lossf = loss.item()

        inner_tq.set_postfix(loss = lossf)
def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )
print(os.listdir('../input/severstal-steel-defect-detection/'))


path = '../input/severstal-steel-defect-detection/'

tr = pd.read_csv(path + 'train.csv')

print(len(tr))

tr.head()
df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)



#df_train1 = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '1')].reset_index(drop=True)

#df_train2 = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '2')].reset_index(drop=True)

#df_train3 = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '3')].reset_index(drop=True)

#df_train4 = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)



#df_train = tr[tr['EncodedPixels']].reset_index(drop=True)

#df_train = tr

print(len(df_train))

df_train.head()
df_train
class ImageData(Dataset):

    def __init__(self, df, transform, subset="train"):

        super().__init__()

        self.df = df

        self.transform = transform

        self.subset = subset

        

        if self.subset == "train":

            self.data_path = path + 'train_images/'

        elif self.subset == "test":

            self.data_path = path + 'test_images/'



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):                      

        fn = self.df['ImageId_ClassId'].iloc[index].split('_')[0]         

        img = Image.open(self.data_path + fn)

        img = self.transform(img)



        if self.subset == 'train': 

            mask = rle2mask(self.df['EncodedPixels'].iloc[index], (256, 1600))

            mask = transforms.ToPILImage()(mask)            

            mask = self.transform(mask)

            return img, mask

        else: 

            mask = None

            return img  
data_transf = transforms.Compose([

                                  transforms.Scale((256, 1600)),

                                  #HorizontalFlip(p=0.5),

                                  #VerticalFlip(p = 0.5),

                                  #Blur(),

                                  #Cutout(),

                                  #ShiftScaleRotate(),

                                  #GaussNoise(),

                                  #ToGray(),

                                  transforms.ToTensor()])

train_data = ImageData(df = df_train, transform = data_transf)
model_ft = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft.to(device)

NUM_GPUS = torch.cuda.device_count()

if NUM_GPUS > 1:

    model_ft = torch.nn.DataParallel(model_ft)

_ = model_ft.to(device)
data_loader = torch.utils.data.DataLoader(

    train_data, batch_size=4, shuffle=True, num_workers=NUM_GPUS,drop_last=True

)
# construct an optimizer

params = [p for p in model_ft.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                               step_size=5,

                                               gamma=0.1)
num_epochs = 2

for epoch in range(num_epochs):

    train_one_epoch(model_ft, optimizer, data_loader, device, epoch)

    lr_scheduler.step()


for param in model_ft.parameters():

    param.requires_grad = False

model_ft.to(torch.device('cuda'))

#assert model_ft.training == False



model_ft.eval()
torch.save(model_ft.state_dict(), 'deeplabv3Resnet101.pth')

torch.cuda.empty_cache()