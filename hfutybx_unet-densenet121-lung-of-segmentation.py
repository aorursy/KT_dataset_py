import matplotlib.pyplot as plt

import numpy as np

import random

import os

import glob

import pandas as pd 

from tqdm import tqdm

import sys

import glob

import cv2



import pydicom

from sklearn.utils import shuffle



import albumentations as A



import torch

import torch.nn as nn

from torch import optim

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

from torch.optim.lr_scheduler import  ReduceLROnPlateau
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch-master')

sys.path.append('../input/pretrainedmodels/pretrainedmodels-0.7.4/')

sys.path.append('../input/segmentation-models-pytorch/')

import segmentation_models_pytorch as smp
root_path = '../input/osiclungmask100'

img_path = sorted(glob.glob(root_path+'/*/img.png'))

mask_path = sorted(glob.glob(root_path+'/*/post_label.png'))



imgpaths,maskpaths = shuffle(img_path,mask_path, random_state=0)



train_images_path = imgpaths[:int(len(imgpaths)*0.8)]

train_masks_path = maskpaths[:int(len(imgpaths)*0.8)]

val_images_path = imgpaths[int(len(imgpaths)*0.8):]

val_masks_path = maskpaths[int(len(maskpaths)*0.8):]



transform = A.Compose([

    A.Rotate(p=0.2,limit=30),

    A.HorizontalFlip(p=0.2),

    A.OneOf([

        A.GridDistortion(p=0.1,distort_limit=0.2),

        A.ElasticTransform(sigma=10, alpha=1,  p=0.1)

    ]),

])

batch = 8

lr = 0.0003

wd = 5e-4

epochs = 80

output_path = './'

device =  torch.device('cuda:0')

experiment_name = 'lung_Unet_densenet121'
class Data_Generate(Dataset):

    def __init__(self,img_paths,seg_paths=None,transform=None):

        self.img_paths = img_paths

        self.seg_paths = seg_paths

        self.transform = transform

        

    def __getitem__(self,index):

        if self.seg_paths is not None:

            img_path = self.img_paths[index]

            mask_path = self.seg_paths[index]

            

            mask = cv2.imread(mask_path,0)/255

            img = cv2.imread(img_path,0)/255



            if self.transform != None:

                aug = transform(image=img,mask=mask)

                img = aug['image']

                mask = aug['mask']

                

            img = img[None,:,:]

            img = img.astype(np.float32)

            mask = mask[None,:,:]

            mask = mask.astype(np.float32)

            

            return img,mask

        

        else:

            img = cv2.imread(self.img_paths[index],0)/255

            img = img[None,:,:]

            img = img.astype(np.float32)

            return img

        

    def __len__(self):

        return len(self.img_paths)
train_db = Data_Generate(train_images_path,train_masks_path,transform=transform)

train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=4)

val_db = Data_Generate(val_images_path,val_masks_path,transform=None)

val_loader = DataLoader(val_db, batch_size=batch, shuffle=False, num_workers=4)
f,ax = plt.subplots(4,4,figsize=(16,16))

for i in range(16):

    img = train_db[i][0]

    ax[i//4,i%4].imshow(img[0])
class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0

            path (str): Path for the checkpoint to be saved to.

                            Default: 'checkpoint.pt'

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta

        self.path = path



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
model = smp.Unet('densenet121', classes=1, in_channels=1,activation='sigmoid',encoder_weights='imagenet').to(device)

    

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8, verbose=True)



criterion = smp.utils.losses.DiceLoss(eps=1.)

iou = smp.utils.metrics.IoU()

early_stopping = EarlyStopping(patience=6, verbose=True,path=os.path.join(output_path, f'best_{experiment_name}.pth'))
num_train_loader = len(train_loader)

num_val_loader = len(val_loader)

for epoch in range(epochs):

    train_losses,train_score,val_losses,val_score = 0,0,0,0

    model.train()



    for idx, sample in enumerate(train_loader):

        image, label = sample

        image, label = image.to(device), label.to(device)

        out = model(image)

        loss = criterion(out, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_losses += loss/num_train_loader

        train_score += iou(out,label)/num_train_loader

    

    model.eval()

    for idx, sample in enumerate(val_loader):

        image, label = sample

        image, label = image.to(device), label.to(device)

        with torch.no_grad():

            out = model(image)

        loss = criterion(out, label)

        val_losses += loss/num_val_loader

        val_score += iou(out,label)/num_val_loader

    print('epoch {}/{}\t LR:{}\t train_loss:{}\t train_score:{}\t val_loss:{}\t val_score:{}' \

          .format(epoch+1, epochs, optimizer.param_groups[0]['lr'], train_losses, train_score, val_losses, val_score))

    scheduler.step(val_losses)

    

    early_stopping(val_losses, model)

    if early_stopping.early_stop:

        print("Early stopping")

        break
class Test_Generate(Dataset):

    def __init__(self,img_paths):

        self.img_paths = img_paths

        

    def __getitem__(self,index):

        dicom = pydicom.dcmread(self.img_paths[index])

        slice_img = dicom.pixel_array

        slice_img = (slice_img-slice_img.min())/(slice_img.max()-slice_img.min())

        slice_img = (slice_img*255).astype(np.uint8)

        if slice_img.shape[0] != 512:

            slice_img = cv2.resize(slice_img,(512,512))

            

        slice_img = slice_img[None,:,:]

        slice_img = (slice_img/255).astype(np.float32)

        return slice_img

        

    def __len__(self):

        return len(self.img_paths)
dicom_root_path = '../input/osic-pulmonary-fibrosis-progression/train/*/*'

dicom_paths = glob.glob(dicom_root_path)

dicom_paths = random.sample(dicom_paths,16)



test_db = Test_Generate(dicom_paths)

test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=0)



model.load_state_dict(torch.load('./best_lung_Unet_densenet121.pth'))

model.eval()



outs = []

for idx, sample in enumerate(test_loader):

    image = sample

    image = image.to(device)

    with torch.no_grad():

        out = model(image)

    out = out.cpu().data.numpy()

    out = np.where(out>0.5,1,0)

    out = np.squeeze(out)

    outs.append(out)

    

outs = np.concatenate(outs)
f,ax = plt.subplots(4,4,figsize=(16,16))

axes = ax.flatten()

for idx in range(len(outs)//2):

    axes[idx*2].imshow(test_db[idx][0])

    axes[idx*2+1].imshow(outs[idx])