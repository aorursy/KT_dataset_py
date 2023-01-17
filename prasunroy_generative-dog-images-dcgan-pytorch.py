# imports

import torch.nn as nn





# weights initialization

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)





# Generator network

class Generator(nn.Module):

    

    def __init__(self, nz=100, nc=3, ngf=64, init_weights=False):

        super(Generator, self).__init__()

        self.network = nn.Sequential(

            # input shape: nz x 1 x 1 -> output shape: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(ngf*8),

            nn.ReLU(inplace=True),

            

            # input shape: (ngf*8) x 4 x 4 -> output shape: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf*4),

            nn.ReLU(inplace=True),

            

            # input shape: (ngf*4) x 8 x 8 -> output shape: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf*2),

            nn.ReLU(inplace=True),

            

            # input shape: (ngf*2) x 16 x 16 -> output shape: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf),

            nn.ReLU(inplace=True),

            

            # input shape: (ngf) x 32 x 32 -> output shape: (nc) x 64 x 64

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),

            nn.Tanh()

        )

        if init_weights:

            self.apply(weights_init)

    

    def forward(self, x):

        return self.network(x)





# Discriminator network

class Discriminator(nn.Module):

    

    def __init__(self, nc=3, ndf=64, init_weights=False):

        super(Discriminator, self).__init__()

        self.network = nn.Sequential(

            # input shape: (nc) x 64 x 64 -> output shape: (ndf) x 32 x 32

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            

            # input shape: (ndf) x 32 x 32 -> output shape: (ndf*2) x 16 x 16

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf*2),

            nn.LeakyReLU(0.2, inplace=True),

            

            # input shape: (ndf*2) x 16 x 16 -> output shape: (ndf*4) x 8 x 8

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf*4),

            nn.LeakyReLU(0.2, inplace=True),

            

            # input shape: (ndf*4) x 8 x 8 -> output shape: (ndf*8) x 4 x 4

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf*8),

            nn.LeakyReLU(0.2, inplace=True),

            

            # input shape: (ndf*8) x 4 x 4 -> output shape: 1 x 1 x 1

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()

        )

        if init_weights:

            self.apply(weights_init)

    

    def forward(self, x):

        return self.network(x)

# imports

import glob

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import shutil

import torch

import torchvision



from PIL import Image

from torch.utils.data import Dataset, DataLoader

from xml.etree import ElementTree

#from dcgan.models import Generator, Discriminator

# configurations

RANDOMSEED = None

IMAGE_ROOT = '../input/all-dogs/all-dogs/'

ANNOT_ROOT = '../input/annotation/Annotation/'

IMAGE_SIZE = 64

N_CHANNELS = 3

LATENT_DIM = 100

BATCH_SIZE = 64

NUM_EPOCHS = 10

LEARN_RATE = 0.0002

USE_N_GPUS = 1

# custom dataset

class CustomDataset(Dataset):

    

    def __init__(self, image_dir, annotation_dir, transform=None):

        self._imgbboxes = {}

        self._transform = transform if not transform is None else torchvision.transforms.ToTensor()

        breed_map = {os.path.basename(breed_dir).split('-')[0]: breed_dir \

                     for breed_dir in glob.glob(f'{annotation_dir}/*')}

        index_img = 0

        for image_path in glob.glob(f'{image_dir}/*.*'):

            breed, index = os.path.splitext(os.path.basename(image_path))[0].split('_')

            for obj in ElementTree.parse(f'{breed_map[breed]}/{breed}_{index}').getroot().findall('object'):

                bbox = obj.find('bndbox')

                xmin = int(bbox.find('xmin').text)

                ymin = int(bbox.find('ymin').text)

                xmax = int(bbox.find('xmax').text)

                ymax = int(bbox.find('ymax').text)

                self._imgbboxes[index_img] = (image_path, (xmin, ymin, xmax, ymax))

                index_img += 1

    

    def __len__(self):

        return len(self._imgbboxes)

    

    def __getitem__(self, index):

        image_path, bbox = self._imgbboxes[index]

        image = Image.open(image_path).crop(bbox)

        image = self._transform(image)

        return {'image': image, 'label': 1}

# show image tensor as grid

def show(x, title=None):

    grid_image = torchvision.utils.make_grid(x, normalize=True).numpy()

    if title:

        plt.title(title)

    plt.axis('off')

    plt.imshow(np.transpose(grid_image, (1, 2, 0)))

    plt.show()

# setup output directory structure

for directory in ['fake', 'ckpt']:

    if not os.path.isdir(f'output/{directory}/'):

        os.makedirs(f'output/{directory}/')





# setup seed for reproducible results

if RANDOMSEED:

    torch.manual_seed(RANDOMSEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False





# setup device

device = torch.device('cuda:0' if torch.cuda.is_available() and USE_N_GPUS > 0 else 'cpu')

device_info = 'CPU' if device.type.lower() == 'cpu' else f'{torch.cuda.get_device_name(device)} [CUDA]'

print(f'[INFO] Using device: {device_info}')

# create transform, dataset and dataloader

if N_CHANNELS == 1:

    transform = torchvision.transforms.Compose([

        torchvision.transforms.Grayscale(),

        torchvision.transforms.Resize(IMAGE_SIZE),

        torchvision.transforms.CenterCrop(IMAGE_SIZE),

        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize((0.5,), (0.5,))

    ])

else:

    transform = torchvision.transforms.Compose([

        torchvision.transforms.Resize(IMAGE_SIZE),

        torchvision.transforms.CenterCrop(IMAGE_SIZE),

        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])



dataset = CustomDataset(IMAGE_ROOT, ANNOT_ROOT, transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# example of real samples

real_batch = next(iter(dataloader))['image'][:64]

torchvision.utils.save_image(real_batch, 'output/real.jpg', normalize=True)

show(real_batch, 'Real samples')

# create generator and discriminator networks

netG = Generator(LATENT_DIM, N_CHANNELS, init_weights=True).to(device)

netD = Discriminator(N_CHANNELS, init_weights=True).to(device)

# create criterion and optimizers

criterion = torch.nn.BCELoss()

optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARN_RATE, betas=(0.5, 0.999))

optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARN_RATE, betas=(0.5, 0.999))

# initialize variables

real_label = 1

fake_label = 0

seed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

loop_width = len(str(NUM_EPOCHS))

step_width = len(str(len(dataloader)))

hist_lossG = []

hist_lossD = []

hist_ckptG = []

hist_ckptD = []





# begin epoch

for epoch in range(NUM_EPOCHS):

    

    

    # iterate over data

    for i, batch in enumerate(dataloader):

        

        

        # ---- train discriminator: maximize log(D(x)) + log(1 - D(G(z))) ----

        netD.zero_grad()

        

        # pass one batch of real images

        real_x = batch['image'].to(device)

        labels = torch.full((real_x.size(0),), real_label, device=device)

        output = netD(real_x).view(-1)

        lossD_real = criterion(output, labels)

        lossD_real.backward()

        D_x = output.mean().item()

        

        # pass one batch of fake images

        z = torch.randn(real_x.size(0), LATENT_DIM, 1, 1, device=device)

        fake_x = netG(z)

        labels.fill_(fake_label)

        output = netD(fake_x.detach()).view(-1)

        lossD_fake = criterion(output, labels)

        lossD_fake.backward()

        D_Gz_1 = output.mean().item()

        

        # estimate total loss over both batches

        lossD = lossD_real + lossD_fake

        

        # update discriminator

        optimizerD.step()

        

        

        # ---- train generator: maximize log(D(G(z))) ----

        netG.zero_grad()

        

        # update generator

        labels.fill_(real_label)

        output = netD(fake_x).view(-1)

        lossG = criterion(output, labels)

        lossG.backward()

        D_Gz_2 = output.mean().item()

        

        optimizerG.step()

        

        

        # ---- record statistics ----

        hist_lossG.append(lossG.item())

        hist_lossD.append(lossD.item())

        step = epoch * len(dataloader) + i

        if step % 500 == 0 or step == NUM_EPOCHS * len(dataloader) - 1:

            print(f'[Epoch {epoch+1:{loop_width}d}/{NUM_EPOCHS}]',

                  f'[Batch {i+1:{step_width}d}/{len(dataloader)}]',

                  f'- Loss_G: {lossG.item():7.4f}',

                  f'- Loss_D: {lossD.item():7.4f}',

                  f'- D(x): {D_x:7.4f}',

                  f'- D(G(z)): {D_Gz_1:7.4f} -> {D_Gz_2:7.4f}')

            with torch.no_grad():

                fake_batch = netG(seed_noise).detach().cpu()

            #torchvision.utils.save_image(fake_batch, f'output/fake/step-{step}.jpg', normalize=True)

            show(fake_batch, f'Fake samples [step: {step}]')

            torch.save(netG.state_dict(), f'output/ckpt/netG-{step}.pt')

            torch.save(netD.state_dict(), f'output/ckpt/netD-{step}.pt')

            hist_ckptG.append(f'output/ckpt/netG-{step}.pt')

            hist_ckptD.append(f'output/ckpt/netD-{step}.pt')

            if len(hist_ckptG) > 5:

                os.remove(hist_ckptG.pop(0))

            if len(hist_ckptD) > 5:

                os.remove(hist_ckptD.pop(0))





# save statistics

df = pd.DataFrame()

df['lossG'] = hist_lossG

df['lossD'] = hist_lossD

df.to_csv('output/statistics.csv', index=False)





# plot statistics

plt.figure(figsize=(10, 5))

plt.plot(hist_lossG, label='G Loss')

plt.plot(hist_lossD, label='D Loss')

plt.xlabel('Iterations')

plt.ylabel('Loss')

plt.title('Generator and Discriminator Loss')

plt.legend()

plt.savefig('output/statistics.jpg')

# prepare submission for Kaggle

print('[INFO] Creating archive for submission... ', end='')

if not os.path.isdir('output/kaggle/'):

    os.makedirs('output/kaggle/')



for i in range(0, 10000, 50):

    z = torch.randn(50, LATENT_DIM, 1, 1, device=device)

    with torch.no_grad():

        fake_x = netG(z).detach().cpu()

    for j in range(50):

        filename = f'output/kaggle/{str(i+j).zfill(4)}.png'

        torchvision.utils.save_image(fake_x[j, :, :, :], filename, normalize=True)



shutil.make_archive('images', 'zip', 'output/kaggle/')

shutil.rmtree('output/kaggle/')

print('done')
