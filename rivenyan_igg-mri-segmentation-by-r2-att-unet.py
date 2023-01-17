import os, cv2

import torch

import torchvision

from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import pandas as pd

from torch.utils.data import DataLoader

import albumentations as album

import torch.nn.init as init

import random

from tqdm import tqdm

from albumentations.pytorch import ToTensor

import torch.nn.functional as F

from torch import optim

from torch.utils.tensorboard import SummaryWriter

import logging
from glob import glob



mask_files = glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')



train_files = []

for i in mask_files:

    train_files.append(i.replace('_mask',''))



for i in range(4):   

    print(mask_files[i])

    print(train_files[i])
def train_augmentation():

    TA = [

        album.OneOf([album.HorizontalFlip(p=1),

                     album.VerticalFlip(p=1),

                     album.RandomRotate90(p=1),], p=5)

    ]

    return album.Compose(TA)

def image_transforms():

    list_transforms=[

        ToTensor(),

    ]

    LT = album.Compose(list_transforms)

    return LT
img_dir = train_files

mask_dir = mask_files

class dataset(torch.utils.data.Dataset):

    def __init__(self, imgpath, maskpath, augmentation=None):

        self.image_paths = imgpath

        self.mask_paths = maskpath

        self.augmentation = augmentation

        self.image_transforms = image_transforms()

    def __getitem__(self, i):

        # image

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]),cv2.COLOR_BGR2RGB)  

        image = cv2.resize(image,(240,240), interpolation=cv2.INTER_AREA)

        # mask

        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i],0),cv2.COLOR_BGR2RGB)

        mask = cv2.resize(mask,(240,240), interpolation =cv2.INTER_NEAREST)

        # augmentation and transforms

        if self.augmentation:

            sample = self.augmentation(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

            okok = self.image_transforms(image=image, mask=mask)

            image, mask = okok['image'], okok['mask']

        return image, mask

    def __len__(self):

        return len(self.image_paths)

    

# Data

train_dataset = dataset(

    img_dir,

    mask_dir,

    augmentation = train_augmentation(),

)



# Dataloader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)



print(train_loader)
def imshow(img):

    if type(img)==torch.Tensor and img.shape[0]==3:

        img=torch.transpose(img,0,1)

        img=torch.transpose(img,2,1)

    elif type(img)==torch.Tensor and img.shape[0]==1:

        img = img[:][:][0]

    print(img.shape)

    plt.imshow(img)

    plt.show()

image_5, mask_5 = train_dataset[5]

imshow(image_5)

imshow(mask_5)
def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):

        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':

                init.normal_(m.weight.data, 0.0, gain)

            elif init_type == 'xavier':

                init.xavier_normal_(m.weight.data, gain=gain)

            elif init_type == 'kaiming':

                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':

                init.orthogonal_(m.weight.data, gain=gain)

            else:

                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:

                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:

            init.normal_(m.weight.data, 1.0, gain)

            init.constant_(m.bias.data, 0.0)



    print('initialize network with %s' % init_type)

    net.apply(init_func)



class conv_block(nn.Module):

    def __init__(self,ch_in,ch_out):

        super(conv_block,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),

            nn.BatchNorm2d(ch_out),

            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),

            nn.BatchNorm2d(ch_out),

            nn.ReLU(inplace=True)

        )





    def forward(self,x):

        x = self.conv(x)

        return x



class up_conv(nn.Module):

    def __init__(self,ch_in,ch_out):

        super(up_conv,self).__init__()

        self.up = nn.Sequential(

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),

		    nn.BatchNorm2d(ch_out),

			nn.ReLU(inplace=True)

        )



    def forward(self,x):

        x = self.up(x)

        return x



class Recurrent_block(nn.Module):

    def __init__(self,ch_out,t=2):

        super(Recurrent_block,self).__init__()

        self.t = t

        self.ch_out = ch_out

        self.conv = nn.Sequential(

            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),

		    nn.BatchNorm2d(ch_out),

			nn.ReLU(inplace=True)

        )



    def forward(self,x):

        for i in range(self.t):



            if i==0:

                x1 = self.conv(x)

            

            x1 = self.conv(x+x1)

        return x1

        

class RRCNN_block(nn.Module):

    def __init__(self,ch_in,ch_out,t=2):

        super(RRCNN_block,self).__init__()

        self.RCNN = nn.Sequential(

            Recurrent_block(ch_out,t=t),

            Recurrent_block(ch_out,t=t)

        )

        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)



    def forward(self,x):

        x = self.Conv_1x1(x)

        x1 = self.RCNN(x)

        return x+x1





class single_conv(nn.Module):

    def __init__(self,ch_in,ch_out):

        super(single_conv,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),

            nn.BatchNorm2d(ch_out),

            nn.ReLU(inplace=True)

        )



    def forward(self,x):

        x = self.conv(x)

        return x



class Attention_block(nn.Module):

    def __init__(self,F_g,F_l,F_int):

        super(Attention_block,self).__init__()

        self.W_g = nn.Sequential(

            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),

            nn.BatchNorm2d(F_int)

            )

        

        self.W_x = nn.Sequential(

            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),

            nn.BatchNorm2d(F_int)

        )



        self.psi = nn.Sequential(

            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),

            nn.BatchNorm2d(1),

            nn.Sigmoid()

        )

        

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self,g,x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(g1+x1)

        psi = self.psi(psi)



        return x*psi

    

class R2AttU_Net(nn.Module):

    def __init__(self,img_ch=3,output_ch=1,t=2):

        super(R2AttU_Net,self).__init__()

        

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)



        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)



        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)

        

        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)

        

        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)

        

        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)

        



        self.Up5 = up_conv(ch_in=1024,ch_out=512)

        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)

        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)

        

        self.Up4 = up_conv(ch_in=512,ch_out=256)

        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)

        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)

        

        self.Up3 = up_conv(ch_in=256,ch_out=128)

        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)

        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)

        

        self.Up2 = up_conv(ch_in=128,ch_out=64)

        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)

        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)



        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)





    def forward(self,x):

        # encoding path

        x1 = self.RRCNN1(x)



        x2 = self.Maxpool(x1)

        x2 = self.RRCNN2(x2)

        

        x3 = self.Maxpool(x2)

        x3 = self.RRCNN3(x3)



        x4 = self.Maxpool(x3)

        x4 = self.RRCNN4(x4)



        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)



        # decoding + concat path

        d5 = self.Up5(x5)

        x4 = self.Att5(g=d5,x=x4)

        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_RRCNN5(d5)

        

        d4 = self.Up4(d5)

        x3 = self.Att4(g=d4,x=x3)

        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_RRCNN4(d4)



        d3 = self.Up3(d4)

        x2 = self.Att3(g=d3,x=x2)

        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_RRCNN3(d3)



        d2 = self.Up2(d3)

        x1 = self.Att2(g=d2,x=x1)

        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_RRCNN2(d2)



        d1 = self.Conv_1x1(d2)



        return d1



print(R2AttU_Net())
def dice_loss(input, target):

    input = torch.sigmoid(input)

    smooth = 1.0

    iflat = input.view(-1)

    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))



def pySigmoidFocalLoss(preds, targets, loss_weight=1.0, gamma=2.0, alpha=0.25, size_average=True, avg_factor=None):

	preds_sigmoid = preds.sigmoid()

	targets = targets.type_as(preds)

	pt = (1 - preds_sigmoid) * targets + preds_sigmoid * (1 - targets)

	focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)

	loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none') * focal_weight



	if avg_factor is None:

		loss = loss.mean() if size_average else loss.sum()

	else:

		loss = (loss.sum() / avg_factor) if size_average else loss.sum()

	return loss * loss_weight   



def MixedLoss(input, target, weight):

    return pySigmoidFocalLoss(input, target, loss_weight=weight) - torch.log(dice_loss(input, target))
def train_net(net, device, epochs=5, lr=0.001, save_cp=True):

    n_val = len(valid_loader)

    n_train = len(train_loader)

    global_step = 0

    logging.info(f'''Starting training:

        Epochs:          {epochs}

        Batch size:      {1}

        Learning rate:   {lr}

        Training size:   {n_train}

        Validation size: {n_val}

        Checkpoints:     {save_cp}

        Device:          {device.type}

    ''')

    #optimizer = optim.Adam(net.parameters(),betas=(.9, 0.999),lr=lr)

    optimizer = optim.Nadam(net.parameters(), lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    for epoch in range(epochs):

        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch{epoch+1}/{epochs}',unit='img')as pbar:

            for batch in train_loader:

                imgs = batch[0]

                true_masks = batch[1]

                imgs = imgs.to(device=device, dtype=torch.float32)

                mask_type = torch.float32

                true_masks = true_masks.to(device=device, dtype=mask_type)

                

                masks_pred = net(imgs)

                loss = MixedLoss(masks_pred, true_masks, 0.25)

                epoch_loss += loss.item()

                

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_value_(net.parameters(), 0.1) # 梯度阈值

                optimizer.step()

                

                pbar.update(imgs.shape[0])

                global_step += 1

                #if global_step % (n_train // (10 * 1))==0:

                #    val_score = eval_net(net, valid_loader, device)

                #    scheduler.step(val_score)

        if (epoch+1)%10 == 0:

            torch.save(net.state_dict(),f'CP_epoch{epoch + 1}.pth')

            print('okok') 
if __name__ == '__main__':

    device = torch.device('cuda')

    net = R2AttU_Net().cuda()

    net.train()

    train_net(net=net,epochs=10,device=device)
net.eval()

img, mask = train_dataset[55]

img = img.unsqueeze(0).cuda()

pred = net(img)

a = torch.sigmoid(pred[0]).cpu().detach().numpy()

a[a>0.5]=1

a[a<0.5]=0

a=torch.from_numpy(a)

print('test a valid img')

imshow(a)

print('This is the real mask')

imshow(mask)