%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import torch

import torchvision

import torchvision.transforms as transforms

import torchvision.datasets as datasets

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms.functional as TF

from torch.utils.data.dataset import Dataset as Dataset

from torch.utils.data import dataloader as DL

from torch.autograd import Variable

from PIL import Image

from glob import glob

import PIL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Assuming that we are on a CUDA machine, this should print a CUDA device:



print(device)
input_folder = '../input'



train_paths = glob('/'.join([input_folder,'train/*.jpg']))

train_masks_paths = glob('/'.join([input_folder,'train_masks/*.gif']))

test_paths = glob('/'.join([input_folder,'test/*.jpg']))



train_paths.sort()

train_masks_paths.sort()

test_paths.sort()



temp1 = train_paths 

train_paths = train_paths[0:1]

temp2 = train_masks_paths

train_masks_paths = train_masks_paths[0:1000]



test_paths = temp1[1100:1300]

test_masks_paths = temp2[1100:1300]



print('Number of training images: ', len(train_paths), 'Number of corresponding masks: ', len(train_masks_paths), 'Number of test images: ', len(test_paths))
class MyDataset(Dataset):

    def __init__(self, image_paths, mask_paths, train=True):

        self.image_paths = image_paths

        self.mask_paths = mask_paths

        

    def transforms(self, image, mask):

        #img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)

        #image = transforms.Resize(size=(64, 64))(image)

        #mask = transforms.Resize(size=(64, 64))(mask)

        image = image.resize((64, 64), PIL.Image.NEAREST)

        mask = mask.resize((64, 64), PIL.Image.NEAREST)

        image = TF.to_tensor(image)

        mask = TF.to_tensor(mask)

        

        return image, mask

    

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])

        mask = Image.open(self.mask_paths[index])

        

        x, y = self.transforms(image, mask)

        return x, y

    

    def __len__(self):

        return len(self.image_paths)
print(len(train_paths))

print(len(train_masks_paths))
train_dataset = MyDataset(train_paths, train_masks_paths)
im, mask = train_dataset[0]

im = im.numpy()

mask = mask.numpy()

prod = np.multiply(im, mask)

print(prod.shape)

plt.imshow(prod[0])
print(im)
print(len(train_dataset))
train_data_loader = DL.DataLoader(train_dataset, batch_size=1, shuffle=False)
print(len(train_data_loader))
def double_conv(in_channels, out_channels):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, 3, padding=1),

        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),

        nn.ReLU(inplace=True)

    )   



class UNet(nn.Module):



    def __init__(self, n_class):

        super().__init__()

                

        self.dconv_down1 = double_conv(3, 64)

        self.dconv_down2 = double_conv(64, 128)

        self.dconv_down3 = double_conv(128, 256)

        self.dconv_down4 = double_conv(256, 512)  



        self.maxpool = nn.MaxPool2d(2)

        

        self.dconv_up3 = double_conv(512, 256)

        self.dconv_up2 = double_conv(256, 128)

        self.dconv_up1 = double_conv(128, 64)

        

        self.TConv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.TConv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.TConv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        

        self.conv_last = nn.Conv2d(64, n_class, 1)

        

    def forward(self, x):

        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)

        

        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)

        

        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)

        

        x = self.dconv_down4(x)

        

        x = self.TConv3(x)

        x = torch.cat([x, conv3], dim=1)



        x = self.dconv_up3(x)

        x = self.TConv2(x)

        x = torch.cat([x, conv2], dim=1)



        x = self.dconv_up2(x)

        x = self.TConv1(x)

        x = torch.cat([x, conv1], dim=1)



        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.sigmoid(out)

        

        return out

model = UNet(n_class=1)

model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=3e-5)

criterion = nn.BCELoss()

model.train()

#Number of parameters in the model

sum([p.numel() for p in model.parameters()])
dtype = torch.cuda.FloatTensor
model.train()



num_epochs = 100

running_loss = 0.0



for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (X, y) in enumerate(train_data_loader):

        X = X.to(device)

        y = y.to(device)

        X = Variable(X)

        y = Variable(y)

        

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, y)

        

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    

    print("loss for epoch " + str(epoch) + ":  " + str(running_loss))
model.eval()



x, y = train_dataset[0]

x = x.unsqueeze(0)

x = x.cuda()

pred = model(x)



pred = pred.squeeze(0)

pred = pred.squeeze(0)



plt.imshow(pred.cpu().detach().numpy())

print(pred)