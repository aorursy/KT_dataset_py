import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

#from google.colab import files

from torch.utils.data import DataLoader

from torchvision import datasets

from torchvision import transforms

from torchvision.utils import save_image

import numpy as np

import datetime

import os, sys

from matplotlib.pyplot import imshow, imsave

%matplotlib inline
MODEL_NAME = 'DCGAN'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_sample_image(G, n_noise):

    """

        save sample 100 images

    """

    z = torch.randn(100, n_noise).to(DEVICE)

    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)

    result = y_hat.cpu().data.numpy()

    img = np.zeros([280, 280])

    for j in range(10):

        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=-1)

    return img
class Discriminator(nn.Module):

    """

        Convolutional Discriminator for MNIST

    """

    def __init__(self, in_channel=1, num_classes=1):

        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(

            # 28 -> 14

            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2),

            # 14 -> 7

            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2),

            # 7 -> 4

            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),

            nn.AvgPool2d(4),

        )

        self.fc = nn.Sequential(

            # reshape input, 128 -> 1

            nn.Linear(128, 1),

            nn.Sigmoid(),

        )

    

    def forward(self, x, y=None):

        y_ = self.conv(x)

        y_ = y_.view(y_.size(0), -1)

        y_ = self.fc(y_)

        return y_
class Generator(nn.Module):

    """

        Convolutional Generator for MNIST

    """

    def __init__(self, input_size=100, num_classes=784):

        super(Generator, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(input_size, 4*4*512),

            nn.ReLU(),

        )

        self.conv = nn.Sequential(

            # input: 4 by 4, output: 7 by 7

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            # input: 7 by 7, output: 14 by 14

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            # input: 14 by 14, output: 28 by 28

            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),

            nn.Tanh(),

        )

        

    def forward(self, x, y=None):

        x = x.view(x.size(0), -1)

        y_ = self.fc(x)

        y_ = y_.view(y_.size(0), 512, 4, 4)

        y_ = self.conv(y_)

        return y_
D = Discriminator().to(DEVICE)

G = Generator().to(DEVICE)
D
transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize(mean=[0.5],

                                std=[0.5])]

)

mnist = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
batch_size = 64
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True)
criterion = nn.BCELoss()

D_opt = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))

G_opt = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
max_epoch = 30 # need more than 20 epochs for training generator

step = 0

n_critic = 1 # for training more k steps about Discriminator

n_noise = 100
D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real

D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake
for epoch in range(max_epoch):

    for idx, (images, labels) in enumerate(data_loader):

        # Training Discriminator

        x = images.to(DEVICE)

        x_outputs = D(x)

        D_x_loss = criterion(x_outputs, D_labels)



        z = torch.randn(batch_size, n_noise).to(DEVICE)

        z_outputs = D(G(z))

        D_z_loss = criterion(z_outputs, D_fakes)

        D_loss = D_x_loss + D_z_loss

        

        D.zero_grad()

        D_loss.backward()

        D_opt.step()



        if step % n_critic == 0:

            # Training Generator

            z = torch.randn(batch_size, n_noise).to(DEVICE)

            z_outputs = D(G(z))

            G_loss = criterion(z_outputs, D_labels)



            D.zero_grad()

            G.zero_grad()

            G_loss.backward()

            G_opt.step()

        

        if step % 500 == 0:

            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))

        

        if step % 1000 == 0:

            G.eval()

            img = get_sample_image(G, n_noise)

            imsave('{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')

            G.train()

        step += 1
# generation to image

G.eval()

imshow(get_sample_image(G, n_noise), cmap='gray')
def save_checkpoint(state, file_name='checkpoint.pth.tar'):

    torch.save(state, file_name)
# Saving params.

# torch.save(D.state_dict(), 'D_c.pkl')

# torch.save(G.state_dict(), 'G_c.pkl')

save_checkpoint({'epoch': epoch + 1, 'state_dict':D.state_dict(), 'optimizer' : D_opt.state_dict()}, 'D_dc.pth.tar')

save_checkpoint({'epoch': epoch + 1, 'state_dict':G.state_dict(), 'optimizer' : G_opt.state_dict()}, 'G_dc.pth.tar')