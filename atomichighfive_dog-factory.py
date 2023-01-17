# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn, functional

import torchvision

from skimage import io, transform, exposure, util

from random import shuffle, seed, choice, random

from collections.abc import Iterable

from tqdm.auto import tqdm

import itertools as it



# Plotting

from plotly.offline import iplot, init_notebook_mode

from plotly import graph_objs as go

init_notebook_mode(connected=True)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import sys



os.makedirs('./output', exist_ok=True)

        

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
class DogLoader(torch.utils.data.Dataset):

    def __init__(self, path, preload_steps = 100, random_batch_size = 32, shuffle_images = True, augment=True, preload_all=False, shuffle_seed = None):

        self.preload_all = preload_all

        self.augment = augment

        self.preload_steps = preload_steps

        self.random_batch_size = random_batch_size

        self.path = path

        self.files = sorted(os.listdir(path))

        self.file_indices = list(range(len(self.files)))

        if shuffle_images:

            if shuffle_seed is not None:

                seed(shuffle_seed)

            shuffle(self.files)

        

        if preload_all:

            self.preload_dict = {}

            for file in tqdm(self.files):

                self.preload_dict[self.path+file] = io.imread(self.path+file)

        

        self.counter = 0

        self.preload = self[0:self.preload_steps]

        self.preload_counter = 0

        

    def __len__(self):

        return len(self.files)

        

    def __iter__(self):

        self.counter = 0

        self.preload = self[0:self.preload_steps]

        self.preload_counter = 0

        return self

    

    def __next__(self):

        if self.counter < len(self):

            if self.preload_steps <= self.preload_counter:

                self.preload = self[self.counter:min(self.counter+self.preload_steps, len(self))]

                self.preload_counter = 0

            self.preload_counter += 1

            self.counter += 1

            return self.preload[self.preload_counter-1]

        else:

            raise StopIteration

                

        raise StopIteration

    

    def __getitem__(self, idx):

        if type(idx) is int:

            return self[idx:idx+1]

        if isinstance(idx, Iterable):

            used_files = [self.files[i] for i in idx]

        else:

            used_files = self.files[idx]

        images = []

        for file in used_files:

            if self.preload_all:

                image = self.preload_dict[self.path+file]

            else:

                image = io.imread(self.path+file)

            w,h,c = image.shape

            if self.augment:

                if random() > 0.5:

                    image = image[:,::-1,:]

                image = transform.rotate(image, -5+10*random(), resize=True)

                image = image[w//2-min(w,h)//2:w//2+min(w,h)//2,h//2-min(w,h)//2:h//2+min(w,h)//2,:]

                image = np.concatenate([exposure.equalize_hist(image[:,:,[i]]) for i in range(3)], axis=2)

            image = transform.resize(image, (64,64))

            images.append(image)

        return torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()

    

    def get_random_batch(self):

        idx = [choice(self.file_indices) for i in range(self.random_batch_size)]

        return self[idx]



def test_DogLoader(length = 30):

    dogs1 = DogLoader("../input/all-dogs/all-dogs/", preload_steps=3, augment=False, shuffle_images=True, shuffle_seed=1)

    dogs2 = DogLoader("../input/all-dogs/all-dogs/", preload_steps=5, augment=False, shuffle_images=True, shuffle_seed=1)

    i = 0

    for image1, image2 in zip(dogs1, dogs2):

        print(f"{dogs1.counter}:{dogs1.preload_counter} {dogs2.counter}:{dogs2.preload_counter}")

        assert (image1==image2).all()

        i += 1

        if i >= length:

            break



test_DogLoader()
data_loader = torch.utils.data.DataLoader(

    dataset=DogLoader(path="../input/all-dogs/all-dogs/"),

    batch_size=32,

    num_workers=4,

    drop_last=True,

    collate_fn=torch.cat

)
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = (4,4), stride = 1, padding = 0, bias = True)

        self.bn0 = nn.BatchNorm2d(128)

        self.act0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (4,4), stride = 2, padding = 0, bias = True)

        self.bn1 = nn.BatchNorm2d(64)

        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (4,4), stride = 2, padding = 0, bias = True)

        self.bn2 = nn.BatchNorm2d(32)

        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (4,4), stride = 2, padding = 0, bias = True)

        self.bn3 = nn.BatchNorm2d(16)

        self.act3 = nn.ReLU()

        self.dense4 = nn.Linear(in_features=400, out_features=1)

        self.act4 = nn.Sigmoid()

        

    def forward(self, x: torch.Tensor):

        x = self.conv0(x)

        x = self.bn0(x)

        x = self.act0(x)

        #print(x.shape)

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.act1(x)

        #print(x.shape)

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.act2(x)

        #print(x.shape)

        x = self.conv3(x)

        x = self.bn3(x)

        x = self.act3(x)

        #print(x.shape)

        x = self.dense4(x.view(-1, 16*5*5))

        x = self.act4(x)

        #print(x.shape)

        return x

    

    

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.dense0 = nn.Linear(in_features = 6, out_features = 16*8*8)

        self.act0 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels= 32, kernel_size=(4,4), stride = 2, padding = 0, bias = True)

        self.bn1 = nn.BatchNorm2d(32)

        self.act1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels= 64, kernel_size=(4,4), stride = 2, padding = 0, bias = True)

        self.bn2 = nn.BatchNorm2d(64)

        self.act2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels= 128, kernel_size=(4,4), stride = 2, padding = 0, bias = True)

        self.bn3= nn.BatchNorm2d(128)

        self.act3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(in_channels = 128, out_channels= 3, kernel_size=(4,4), stride = 1, padding = 1, bias = True)

        self.bn4= nn.BatchNorm2d(3)

        self.act4= nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        x = self.dense0(x)

        x = self.act0(x)

        #print(x.shape)

        x = self.deconv1(x.view(-1, 16, 8, 8))

        x = self.bn1(x)

        x = self.act1(x)

        #print(x.shape)

        x = self.deconv2(x)

        x = self.bn2(x)

        x = self.act2(x)

        #print(x.shape)

        x = self.deconv3(x)

        x = self.bn3(x)

        x = self.act3(x)

        #print(x.shape)

        x = self.deconv4(x)

        x = self.bn4(x)

        x = self.act4(x)

        #print(x.shape)

        x = x[:, :, 7:71, 7:71]

        #print(x.shape)

        return x
LR_g = 0.0002

LR_d = 0.0002

mu = 1

gamma = 0.001

criterion = nn.BCELoss()



discriminator = Discriminator().cuda()

generator = Generator().cuda()



optimizerD = torch.optim.Adam(discriminator.parameters(), lr=LR_d, betas=(0.5, 0.999))

optimizerG = torch.optim.Adam(generator.parameters(), lr=LR_g, betas=(0.5, 0.999))



steps = 500

outputs = []

with tqdm(total = steps) as pbar:

    for i, real_images in enumerate(data_loader):

        if i > steps:

            break

        real_images = real_images.cuda()

        for step in range(20):

            noise = torch.normal(torch.zeros((real_images.shape[0],6)),1.0).cuda()

            fake_images = generator.forward(noise).detach().cuda()

            input_images = torch.cat([real_images, fake_images], dim=0).cuda()

            target = torch.cat([torch.ones(real_images.shape[0], 1), torch.zeros(fake_images.shape[0], 1)], dim=0).cuda()



            # Update discriminator

            discriminator.zero_grad()

            loss = criterion(discriminator.forward(input_images), target)

            loss.backward()

            optimizerD.step()



            # Update generator

            generator.zero_grad()

            generated = generator.forward(noise)

            dispersion = (torch.cat([(a.flatten()-b.flatten())**16 for a, b in it.product(generated, generated)]).sum())**(1.0/16.0)

            loss = criterion(discriminator.forward(torch.cat([real_images, generator.forward(noise)], dim=0)), torch.zeros_like(target))\

            - mu*torch.sigmoid(gamma*dispersion)

            loss.backward()

            optimizerG.step()

        

        pbar.set_postfix({'Dispersion':float(dispersion)})

            

        # Save images

        if i % 5 == 0:

            outputs.append(fake_images.cpu().permute(0,2,3,1).detach().numpy())

        pbar.update(1)
for step in outputs:

    io.imshow_collection(step[0:2])
for i in range(outputs[-1].shape[0]):

    if i %3 == 0:

        io.imshow_collection(outputs[-1][i:i+2])