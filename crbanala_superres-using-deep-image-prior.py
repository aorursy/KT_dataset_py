# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torchvision.utils import make_grid

import torchvision.utils as vutils

import matplotlib.animation as animation

from IPython.display import HTML



import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import pandas as pd

import copy

import time

import cv2 as cv

from tqdm import tqdm_notebook as tqdm

import matplotlib.image as mpimg









import torchvision.transforms.functional as TF



transform = transforms.Compose([transforms.ToTensor(),])

crop = transforms.CenterCrop((240,240))

hrscale = transforms.Scale((160,256))

lrscale = transforms.Scale((40,64))



img = mpimg.imread("/kaggle/input/tiger.jpg")

print(img.shape)

img = (TF.to_pil_image(img))





hrimg = (transform(hrscale(img)) -0.5) /0.5

lrimg = (transform(lrscale(img)) -0.5) /0.5
hrimgv = hrimg *0.5 + 0.5



lrimgv = lrimg*0.5 + 0.5



f, axarr = plt.subplots(1,2)

axarr[0].title.set_text('Original \n Image')

axarr[1].title.set_text('DownSampled Image')



axarr[0].imshow(hrimgv.permute(1,2,0))

axarr[1].imshow(lrimgv.permute(1,2,0))

f.set_figheight(11)

f.set_figwidth(11)

plt.show()
#https://dmitryulyanov.github.io/deep_image_prior
# Generator / Decoder Model

num_channels_in_encoder = 8

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        

        # DECODER

#         self.latent_fc1 = nn.Sequential(

#             nn.Linear(latent_size,1000),

#             nn.Sigmoid(),

#         )

#         self.latent_fc2 = nn.Sequential(

#             nn.Linear(1000,54*44),

#             nn.Sigmoid(),

#         )

        # 128x64x64

        self.d_up_conv_1 = nn.Sequential(

        nn.Conv2d(in_channels=num_channels_in_encoder, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),

            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))

        )



        # 128x64x64

        self.d_block_1 = nn.Sequential(

            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),



            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

        )



        # 128x64x64

        self.d_block_2 = nn.Sequential(

            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),



            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

        )



        # 128x64x64

        self.d_block_3 = nn.Sequential(

            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),



            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),

        )



        # 256x128x128

        self.d_up_conv_2 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),



            nn.ZeroPad2d((1, 1, 1, 1)),

            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))

        )



        # 3x128x128

        self.d_up_conv_3 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),

            nn.Sigmoid(),



            nn.ReflectionPad2d((3, 3, 3, 3)),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),

            nn.Tanh()

        )

        

        

        self.downsample = nn.AvgPool2d((4,4))

        self.downsample1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2))

        self.downsample2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2))



        self.image = None

        

    def forward(self, x):

        uc1 = self.d_up_conv_1(x)

        dblock1 = self.d_block_1(uc1) + uc1

        dblock2 = self.d_block_2(dblock1) + dblock1

        dblock3 = self.d_block_3(dblock2) + dblock2

        uc2 = self.d_up_conv_2(dblock3)

        dec = self.d_up_conv_3(uc2)

        self.image = dec

#         print(dec.shape)

#         dec = F.interpolate(dec,size=(40,64), mode='bilinear')

        dec = self.downsample(dec)

#         dec = self.downsample2(dec)

#         print(dec.shape)

        return dec

# custom weights initialization called on netG and netD

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)

device = 'cuda'
z_size = [num_channels_in_encoder,39,63]



netG = Generator().to(device)

netG.apply(weights_init)

# inp = torch.randn(1*z_size[0]*z_size[1]*z_size[2]).view((-1,z_size[0],z_size[1],z_size[2])).to(device)

# output = netG(inp)

# print(output.shape)

# #218 * 178

# del inp

# del output

# torch.cuda.empty_cache()

fixed_inp = torch.randn(1*z_size[0]*z_size[1]*z_size[2]).view((-1,z_size[0],z_size[1],z_size[2])).to(device)
lr = 0.00002

# Initialize BCELoss function

criterion = nn.BCELoss()

msecriterion = nn.MSELoss()

l1criterion = nn.L1Loss()



optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))



# optimizerG = optim.SGD(netG.parameters(), lr=lr)
hrimgs = hrimg.view(-1,hrimg.shape[0],hrimg.shape[1],hrimg.shape[2]).to(device)

lrimgs = lrimg.view(-1,lrimg.shape[0],lrimg.shape[1],lrimg.shape[2]).to(device)
print("Starting prior training loop...")

num_epochs = 160000

index = 0

# For each epoch









for epoch in range(num_epochs):

    netG.train()

    netG.zero_grad()

    output = netG(fixed_inp)

#     print(output.shape)



    # now compute loss and backpropogate

    optimizerG.zero_grad()

#     output = F.interpolate(output,size=(40,64), mode='bicubic')

#     loss = msecriterion(loutput, lrimgs)

    

    loss = msecriterion(output, lrimgs) + 2 * l1criterion(output, lrimgs)

    

#     print(loss.item())

    loss.backward()

    optimizerG.step()



    

    if index % 10000 ==0:

        netG.eval()

        print(loss.item())

        showimg = (netG.image[0].cpu().detach().permute(1, 2, 0)) * 0.5 + 0.5

        f, axarr = plt.subplots(1)

        axarr.imshow(showimg)

        plt.show()

    

    index = index + 1
netG.eval()

showimg = (netG.image[0].cpu().detach().permute(1, 2, 0)) * 0.5 + 0.5

f, axarr = plt.subplots(1,3)





axarr[0].title.set_text('Original \n Image')

axarr[1].title.set_text('DownSampled Image')

axarr[2].title.set_text('Generated Image')







axarr[0].imshow(hrimgv.permute(1,2,0))

axarr[1].imshow(lrimgv.permute(1,2,0))

axarr[2].imshow(showimg)





f.set_figheight(25)

f.set_figwidth(25)



plt.show()
showimg = (output[0].cpu().detach().permute(1, 2, 0)) * 0.5 + 0.5

plt.imshow(showimg)
random_inp = torch.randn(1*z_size[0]*z_size[1]*z_size[2]).view((-1,z_size[0],z_size[1],z_size[2])).to(device)

netG.eval()

showimg = (netG(random_inp)[0].cpu().detach().permute(1, 2, 0)) * 0.5 + 0.5

plt.imshow(showimg)