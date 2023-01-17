# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.image as mpimg

import time
kernel_start_time = time.perf_counter()
# Redefine print to output logs during commit

__print__ = print

def print(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
import torch

from torch import nn

from torch import optim

from torch import utils

from tqdm import tqdm

import torch.nn.functional as F

from torchvision.utils import save_image

from torchvision import datasets, transforms
DATA_DIR = '../input/all-dogs/'

IMG_SIZE = 64

WORKERS = 8

BATCH_SIZE = 64

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

REAL_LABEL = 0.9

FAKE_LABEL = 0.1

NZ = 100

NZ_SHAPE = [100, 1, 1]

NGF = 64

NDF = 64

NC = 3

LRG = 1e-3

LRD = 5e-4

BETAS = (0.5, 0.999)

EPOCHS = 800

SHOW_EVERY = 50

IMGS_TO_DISPLAY = 8

OUTPUT_DIR = 'output_images_1'
DEVICE
def show_generated_img(netG, noise):

    sample = []

    gen_images = netG(noise).to("cpu").clone().detach()

    for tensor in gen_images:

        gen_image = tensor.numpy().transpose(1, 2, 0)

        sample.append(gen_image)



    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))

    for index, axis in enumerate(axes):

        axis.axis('off')

        image_array = (sample[index] + 1.) / 2.

        axis.imshow(image_array)

    plt.show()
def pbar_desc(epoch, dloss, gloss):

    return f'{epoch:04d}/{EPOCHS} | D: {dloss:.3f}  G:{gloss:.3f}'
class Conv2dDCGAN(nn.Module):

    def __init__(self, inf, outf, kernel_size, stride, padding, bias):

        super().__init__()

        self.conv = nn.Conv2d(inf, outf, kernel_size, stride, padding, bias=bias)

        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)



    def forward(self, x):

        return self.conv(x)





class BatchNorm2dDCGAN(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.bn = nn.BatchNorm2d(features)

        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)

        nn.init.constant_(self.bn.bias.data, 0.0)



    def forward(self, x):

        return self.bn(x)

    

class UpscaleBlockLRelu(nn.Module):

    def __init__(self, inf, outf, ksz, st, pad, upscale):

        super().__init__()



        self.conv = Conv2dDCGAN(inf, outf, ksz, st, pad, bias=False)

        self.bn = BatchNorm2dDCGAN(outf // upscale ** 2)

        self.ps = nn.PixelShuffle(upscale)

        self.prelu = nn.LeakyReLU(negative_slope=0.05)



    def forward(self, x):

        x = self.conv(x)

        x = self.ps(x)

        x = self.bn(x)

        x = self.prelu(x)

        return x

    

class DownsampleBlock(nn.Module):

    def __init__(self, inf, outf, ksz, st, pad):

        super().__init__()



        self.conv = Conv2dDCGAN(inf, outf, ksz, st, pad, bias=False)



        self.bn = BatchNorm2dDCGAN(outf)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = F.leaky_relu(x, 0.2, inplace=True)



        return x
class Generator(nn.Module):

    def __init__(self, nz, ngf, nc, block_cls):

        super().__init__()



        self.b1 = block_cls(nz, ngf * 8 * 16, 1, 1, 0, 4)  # Upscale by 4

        self.b2 = block_cls(ngf * 8, ngf * 4 * 4, 3, 1, 1, 2) # Upscale by 2

        self.b3 = block_cls(ngf * 4, ngf * 2 * 4, 3, 1, 1, 2) # Upscale by 2

        self.b4 = block_cls(ngf * 2, ngf * 4, 3, 1, 1, 2)  # Upscale by 2

        self.b5 = block_cls(ngf, ngf * 4, 3, 1, 1, 2) # Upscale by 2



        self.conv = Conv2dDCGAN(ngf, nc, 3, 1, 1, bias=False)



    def forward(self, x):

        x = self.b1(x)   # (ngf x 8) x 4 x 4

        x = self.b2(x)   # (ngf x 4) x 8 X 8

        x = self.b3(x)   # (ngf x 2) x 16 x 16

        x = self.b4(x)   # ngf x 32 X 32

        x = self.b5(x)   # ngf X 64 x 64

        x = self.conv(x)

        return torch.tanh(x)
class Discriminator(nn.Module):

    def __init__(self, nc, ndf, sigmoid=True):

        super().__init__()



        self.start_conv = Conv2dDCGAN(nc, ndf, 4, 2, 1, bias=False)



        self.b1 = DownsampleBlock(ndf, ndf * 2, 3, 2, 1)

        self.b2 = DownsampleBlock(ndf * 2, ndf * 4, 3, 2, 1)

        self.b3 = DownsampleBlock(ndf * 4, ndf * 8, 3, 2, 1)



        self.end_conv = Conv2dDCGAN(ndf * 8, 1, 4, 1, 0, bias=False)

        self.sigmoid = sigmoid



    def forward(self, x):

        x = self.start_conv(x)  # ndf x 32 x 32

        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.b1(x)    # (ndf x 2) x 16 x 16

        x = self.b2(x)    # (ndf x 4) x 8 x 8

        x = self.b3(x)    # (ndf x 8) x 4 x 4

        x = self.end_conv(x)  # 1 x 1 x 1

        x = x.view(x.size(0))



        if not self.sigmoid:

            return x

        return torch.sigmoid(x)
ds = datasets.ImageFolder(root=DATA_DIR,

                            transform=transforms.Compose([

                                transforms.Resize(IMG_SIZE),

                                transforms.CenterCrop(IMG_SIZE),

                                transforms.ToTensor(),

                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

                                ])

                            )



dl = utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
G = Generator(NZ, NGF, NC, UpscaleBlockLRelu)

D = Discriminator(NC, NDF, sigmoid=False)



G.to(DEVICE)

D.to(DEVICE)



optG = optim.Adam(G.parameters(), lr=LRG, betas=BETAS)

optD = optim.Adam(D.parameters(), lr=LRD, betas=BETAS)



# criterion = nn.BCELoss().to(DEVICE)

start_epoch = 1
fixed_noise = torch.randn(*([IMGS_TO_DISPLAY] + NZ_SHAPE)).to(DEVICE)
for epoch in range(start_epoch, EPOCHS + 1):

    

    if time.perf_counter() - kernel_start_time > 32000:

            print("Time limit reached! Stopping kernel!")

            break

            

    # pbar = tqdm(dl, desc=pbar_desc(epoch, 0.0, 0.0))

    print(f'Epoch {epoch}')

    losses = []

    for batch, _ in dl:



        batch = batch.to(DEVICE)

        optD.zero_grad()



        # Train the discriminator

        for param in D.parameters():

            param.requires_grad = True



        # Train on real images

        num_images = batch.size(0)

        rlabels = torch.full((num_images,), REAL_LABEL).to(DEVICE)

        real_d_predictions = D(batch)

        # d_real_loss = criterion(real_d_predictions, rlabels)



        # Train on fake images

        z = torch.randn(*([num_images] + NZ_SHAPE)).to(DEVICE)

        fake = G(z).detach()

        # flabels = torch.full((num_images,), FAKE_LABEL).to(DEVICE)

        fake_d_predictions = D(fake)

        # d_fake_loss = criterion(fake_d_predictions, flabels)



        d_loss = (torch.mean((real_d_predictions - torch.mean(fake_d_predictions) - rlabels) ** 2) +

                  torch.mean((fake_d_predictions - torch.mean(real_d_predictions) + rlabels) ** 2))/2

        d_loss.backward(retain_graph=True)



        optD.step()



        losses.append(d_loss.item())



        # Train the generator

        for param in D.parameters():

            param.requires_grad = False



        optG.zero_grad()



        rlabels = torch.full((num_images,), REAL_LABEL).to(DEVICE)

        z = torch.randn(*([num_images] + NZ_SHAPE)).to(DEVICE)

        fake = G(z)

        predictions = D(fake)

        g_loss = (torch.mean((real_d_predictions - torch.mean(predictions) + rlabels) ** 2) +

                  torch.mean((predictions - torch.mean(real_d_predictions) - rlabels) ** 2))/2

        g_loss.backward()

        optG.step()

        # pbar.set_description(pbar_desc(epoch, d_loss.item(), g_loss.item()))



    # Display generated images 



#     if epoch % SHOW_EVERY == 0 or epoch == 1:

#         show_generated_img(G, fixed_noise)
import shutil

if os.path.exists(f'../{OUTPUT_DIR}'):

    shutil.rmtree(f'../{OUTPUT_DIR}')

os.mkdir(f'../{OUTPUT_DIR}')

im_batch_size = 50

n_images=10000

print(f'Generating {n_images} images')

for i_batch in tqdm(range(0, n_images, im_batch_size)):

    gen_z = torch.randn(*([im_batch_size] + NZ_SHAPE)).to(DEVICE)

    gen_images = G(gen_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join(f'../{OUTPUT_DIR}', f'image_{i_batch+i_image:05d}.png'))





shutil.make_archive('images', 'zip', f'../{OUTPUT_DIR}')