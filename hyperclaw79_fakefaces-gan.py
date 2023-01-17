!pip install -U nvidia-ml-py3
from __future__ import print_function

import os

import random

import time

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.optim as optim

import torch.utils.data

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils

from torch.autograd import Variable

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tqdm.notebook import tqdm



torch.cuda.empty_cache()

torch.cuda.set_device("cuda:0")

torch.backends.cudnn.benchmark = True

torch.backends.cudnn.enabled = True



from pynvml import *

nvmlInit()

deviceCount = nvmlDeviceGetCount()

for i in range(deviceCount):

    handle = nvmlDeviceGetHandleByIndex(i)

    print(f"Device {i} {nvmlDeviceGetName(handle).decode()}")

info = nvmlDeviceGetMemoryInfo(handle)    

def gpu_stats(message=''):

    message = f"{message}\n" or ''

    tqdm.write(

        f"{message}"

        f"Free GPU memory - {info.free // (1024 * 1024)} MB\n"

        f"Used GPU memory - {info.used // (1024 * 1024)} MB\n"

    )

gpu_stats()
%%time

torch.manual_seed(70)

batch_size = 64

image_size = 64



random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]

transform = transforms.Compose([transforms.Resize(image_size),

                                transforms.CenterCrop(image_size),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomApply(random_transforms, p=0.2),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder('../input/', transform=transform)

dataloader = torch.utils.data.DataLoader(

    train_data, shuffle=True,

    batch_size=batch_size,

    pin_memory=True

)

imgs, label = next(iter(dataloader))

imgs = imgs.numpy().transpose(0, 2, 3, 1)
for i in range(5):

    plt.imshow(imgs[i])

    plt.show()
def weights_init(m):

    """

    Takes as input a neural network m that will initialize all its weights.

    """

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)
batch_size = 64

LR_G = 0.0005

LR_D = 0.001



beta1 = 0.5

epochs = 20



real_label = 0.9

fake_label = 0

nz = 128
class Discriminator(nn.Module):

    def __init__(self, channels=3, features_d=64):

        super(Discriminator, self).__init__()

        

        self.channels = channels



        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):

            block = [

                nn.Conv2d(

                    n_input,

                    n_output,

                    kernel_size=k_size,

                    stride=stride,

                    padding=padding,

                    bias=False

                )

            ]

            if bn:

                block.append(nn.BatchNorm2d(n_output))

            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block



        self.model = nn.Sequential(

            *convlayer(self.channels, features_d, 4, 2, 1),

            *convlayer(features_d, features_d * 2, 4, 2, 1),

            *convlayer(features_d * 2, features_d * 4, 4, 2, 1, bn=True),

            *convlayer(features_d * 4, features_d * 8, 4, 2, 1, bn=True),

            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),  # FC with Conv.

        )



    def forward(self, imgs):

        logits = self.model(imgs)

        out = torch.sigmoid(logits)

    

        return out.view(-1, 1)

    



class Generator(nn.Module):

    def __init__(self, nz=256, channels=3):

        super(Generator, self).__init__()

        

        self.nz = nz

        self.channels = channels

        

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):

            block = [

                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),

                nn.BatchNorm2d(n_output),

                nn.ReLU(inplace=True),

            ]

            return block



        self.model = nn.Sequential(

            *convlayer(self.nz, 1024, 4, 1, 0), # Fully connected layer via convolution.

            *convlayer(1024, 512, 4, 2, 1),

            *convlayer(512, 256, 4, 2, 1),

            *convlayer(256, 128, 4, 2, 1),

            *convlayer(128, 64, 4, 2, 1),

            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),

            nn.Tanh()

        )



    def forward(self, z):

        z = z.view(-1, self.nz, 1, 1)

        img = self.model(z)

        return img



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netD = Discriminator().to(device)

netG = Generator(nz).to(device)



criterion = nn.BCELoss()



optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))



if os.path.exists('../working/discriminator.pth'):

    tqdm.write("Found checkpoint, preloading the data...")

    #Disc

    checkpoint = torch.load('../working/discriminator.pth')

    netD.load_state_dict(checkpoint['model_state_dict'])

    optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = epochs - checkpoint['epoch']

    errD = checkpoint['loss']

    # Gen

    checkpoint = torch.load('../working/generator.pth')

    netG.load_state_dict(checkpoint['model_state_dict'])

    optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = epochs - checkpoint['epoch']

    errG = checkpoint['loss']



G_losses = []

D_losses = []

epoch_time = []
def plot_loss (G_losses, D_losses, epoch):

    plt.figure(figsize=(10,5))

    plt.title("Generator and Discriminator Loss - EPOCH "+ str(epoch))

    plt.plot(G_losses,label="G")

    plt.plot(D_losses,label="D")

    plt.xlabel("iterations")

    plt.ylabel("Loss")

    plt.legend()

    plt.show()

    

def show_generated_img(n_images=5):

    sample = []

    for _ in range(n_images):

        noise = torch.randn(1, nz, 1, 1, device=device)

        gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)

        gen_image = gen_image.numpy().transpose(1, 2, 0)

        sample.append(gen_image)



    figure, axes = plt.subplots(1, len(sample), figsize = (64,64))

    for index, axis in enumerate(axes):

        axis.axis('off')

        image_array = sample[index]

        axis.imshow(image_array)



    plt.show()

    plt.close()
%%time

for epoch in tqdm(range(epochs), position=1):

    start = time.time()

    for ii, (real_images, train_labels) in tqdm(

        enumerate(dataloader),

        total=len(dataloader),

        position=0,

        leave=False

    ):

        ############################

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        ###########################

        # train with real

        netD.zero_grad()

        real_images = real_images.to(device)

        train_labels = train_labels.to(device)

        batch_size = real_images.size(0)

        labels = torch.full((batch_size, 1), real_label, device=device)

        labels = labels.to(device)

        

        output = netD(real_images)

        output = output.to(device)

        errD_real = criterion(output, labels)

        errD_real.to(device)

        errD_real.backward()

        D_x = output.mean().item()

        

        # train with fake

        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        noise = noise.to(device)

        fake = netG(noise)

        fake = fake.to(device)

        labels.fill_(fake_label)

        output = netD(fake.detach())

        output.to(device)

        errD_fake = criterion(output, labels)

        errD_fake = errD_fake.to(device)

        errD_fake.backward()

        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        

        ############################

        # (2) Update G network: maximize log(D(G(z)))

        ###########################

        netG.zero_grad()

        labels.fill_(real_label)  # fake labels are real for generator cost

        output = netD(fake)

        output = output.to(device)

        errG = criterion(output, labels)

        errG.to(device)

        errG.backward()

        D_G_z2 = output.mean().item()

        optimizerG.step()

        

        # Save Losses for plotting later

        G_losses.append(errG.item())

        D_losses.append(errD.item())

        

        if (ii+1) % (len(dataloader)//2) == 0:

            tqdm.write(

                f'[{epoch + 1}/{epochs}][{ii+1}/{len(dataloader)}] '

                f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '

                f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}'

            )

            gpu_stats(message=f"Stats for iteration {ii+1}:\n")

    plot_loss (G_losses, D_losses, epoch)

    G_losses = []

    D_losses = []

    show_generated_img()

    torch.save(

        {

            'epoch': epoch,

            'model_state_dict': netD.state_dict(),

            'optimizer_state_dict': optimizerD.state_dict(),

            'loss': errD

        },

        '../working/discriminator.pth'

    )

    torch.save(

        {

            'epoch': epoch,

            'model_state_dict': netG.state_dict(),

            'optimizer_state_dict': optimizerG.state_dict(),

            'loss': errG

        },

        '../working/generator.pth'

    )

    epoch_time.append(time.time()- start)



#fixed_noise = torch.randn(25, nz, 1, 1, device=device)    

#valid_image = netG(fixed_noise)
print (">> average EPOCH duration = ", np.mean(epoch_time))
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

    

im_batch_size = 50

n_images=10000



for i_batch in tqdm(range(0, n_images, im_batch_size)):

    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)

    gen_images = netG(gen_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
fig = plt.figure(figsize=(25, 16))

# display 10 images from each class

for i, j in enumerate(images[:32]):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    plt.imshow(j)
import shutil

shutil.make_archive('images', 'zip', '../output_images')