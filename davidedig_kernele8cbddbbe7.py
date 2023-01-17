import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# First, look at everything.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Imports

%matplotlib inline

import torch

from torch.utils.data import DataLoader

import torchvision

from torchvision import transforms

from torchvision import datasets

import torch.nn as nn

import torch.nn.functional as F

import torch.optim

from matplotlib import pyplot as plt

from matplotlib.pyplot import figure
# Setup data transforms

load_size = 144

crop_size = 128

mean = 0.5

std = 0.5

train_transform = transforms.Compose([

    transforms.Resize(load_size),

    transforms.RandomCrop(crop_size),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((mean,mean,mean), (std,std,std))

])

# Load CIFAR-10 dataset

# Get dataset from https://drive.google.com/file/d/1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF/view

dataset = datasets.ImageFolder(root = "../input", transform = train_transform)

# Create loaders

batch_size = 64

loader = DataLoader(dataset, batch_size=batch_size, shuffle = True,  num_workers=4, pin_memory=True, drop_last=True)
# Custom weight initialization

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)
# DCGAN generator from WGAN-GP code

class Generator(nn.Module):

    

    def __init__(self, nz=100, base_filters=128):

        # Parent construct

        super().__init__()

        # Alias for base filters

        F = base_filters

        self.F = F

        # Layers

        self.lin_1 = nn.Linear(nz, 4*4*4*F)

        self.bn_1 = nn.BatchNorm1d(4*4*4*F)

        self.convt_2 = nn.ConvTranspose2d(4*F, 3*F, kernel_size=4, padding=1, stride=2)

        self.bn_2 = nn.BatchNorm2d(3*F)

        self.convt_3 = nn.ConvTranspose2d(3*F, 3*F, kernel_size=4, padding=1, stride=2)

        self.bn_3 = nn.BatchNorm2d(3*F)

        self.convt_4 = nn.ConvTranspose2d(3*F, 2*F, kernel_size=4, padding=1, stride=2)

        self.bn_4 = nn.BatchNorm2d(2*F)

        self.convt_5 = nn.ConvTranspose2d(2*F, F, kernel_size=4, padding=1, stride=2)

        self.bn_5 = nn.BatchNorm2d(F)

        self.convt_6 = nn.ConvTranspose2d(F, 3, kernel_size=4, padding=1, stride=2)



    # Input: BxZ

    def forward(self, x):

        # Layer 1

        x = F.relu(self.bn_1(self.lin_1(x)))

        x = x.view(-1, 4*self.F, 4, 4)

        x = F.relu(self.bn_2(self.convt_2(x)))

        x = F.relu(self.bn_3(self.convt_3(x)))

        x = F.relu(self.bn_4(self.convt_4(x)))

        x = F.relu(self.bn_5(self.convt_5(x)))

        x = torch.tanh(self.convt_6(x))

        return x
# DCGAN discriminator from WGAN-GP code with BN at first layer

class Discriminator(nn.Sequential):

    

    def __init__(self, base_filters=128):

        # Parent construct

        super().__init__()

        # Alias for base filters

        F = base_filters

        self.F = F

        # Layers

        self.conv_1 = nn.Conv2d(3, F, kernel_size=5, padding=2, stride=2)

        self.bn_1 = nn.BatchNorm2d(F)

        self.conv_2 = nn.Conv2d(F, 2*F, kernel_size=5, padding=2, stride=2)

        self.bn_2 = nn.BatchNorm2d(2*F)

        self.conv_3 = nn.Conv2d(2*F, 3*F, kernel_size=5, padding=2, stride=2)

        self.bn_3 = nn.BatchNorm2d(3*F)

        self.conv_4 = nn.Conv2d(3*F, 3*F, kernel_size=5, padding=2, stride=2)

        self.bn_4 = nn.BatchNorm2d(3*F)

        self.conv_5 = nn.Conv2d(3*F, 4*F, kernel_size=5, padding=2, stride=2)

        self.bn_5 = nn.BatchNorm2d(4*F)

        self.lin_6 = nn.Linear(4*4*4*F, 1)



    # Input: Bx3xHxW

    def forward(self, x):

        x = F.leaky_relu(self.bn_1(self.conv_1(x)), 0.2)

        x = F.leaky_relu(self.bn_2(self.conv_2(x)), 0.2)

        x = F.leaky_relu(self.bn_3(self.conv_3(x)), 0.2)

        x = F.leaky_relu(self.bn_4(self.conv_4(x)), 0.2)

        x = F.leaky_relu(self.bn_5(self.conv_5(x)), 0.2)

        x = x.view(-1, 4*4*4*self.F)

        x = torch.sigmoid(self.lin_6(x))

        x = x.view(-1)

        return x
# Load checkpoint

start_epoch = 40

checkpoint = f"checkpoint-{start_epoch}.pth"

use_checkpoint = True
# Check checkpoint

if use_checkpoint:

    loaded = torch.load(checkpoint)

else:

    start_epoch = 0



# Create generator model/optimizer

g_net = Generator(nz=128, base_filters=128)

if use_checkpoint:

    g_net.load_state_dict(loaded["g_state_dict"])

else:

    g_net.apply(weights_init)

g_optimizer = torch.optim.Adam(g_net.parameters(), lr=0.0002, weight_decay=5e-4, betas=(0.5,0.999))



# Create discriminator model/optimizer

d_net = Discriminator(base_filters=128)

if use_checkpoint:

    d_net.load_state_dict(loaded["d_state_dict"])

else:

    d_net.apply(weights_init)

d_optimizer = torch.optim.Adam(d_net.parameters(), lr=0.0002, weight_decay=5e-4, betas=(0.5,0.999))



# Setup device

dev = torch.device("cuda")

g_net.to(dev);

d_net.to(dev);
# Auxiliary variables

noise = torch.FloatTensor(batch_size, 128).to(dev)

target = torch.FloatTensor(batch_size).to(dev)



# Checkpoint options

save_every = 1



# Training mode

d_net.train()

g_net.train()

# Start training

for epoch in range(start_epoch+1, 100):

    # Loss measures

    sum_g_loss = 0; num_g_loss = 0;

    sum_d_real_loss = 0; num_d_real_loss = 0;

    sum_d_fake_loss = 0; num_d_fake_loss = 0;

    # Get data iterator

    data_iter = iter(loader)

    data_len = len(loader)

    data_i = 0

    # Process all training batches

    while data_i < data_len:

        # Compute gradients for discriminator

        for p in d_net.parameters(): p.requires_grad = True

        

        # Read data

        (real_input, labels) = next(data_iter)

        real_input = real_input.to(dev)

        labels = labels.to(dev)

        data_i += 1



        # Reset discriminator gradients

        d_optimizer.zero_grad()



        # Forward (discriminator, real)

        target.fill_(1)

        target_v = target.detach()

        output = d_net(real_input)

        # Compute loss (discriminator, real)

        d_real_loss = F.binary_cross_entropy(output, target_v)

        sum_d_real_loss += d_real_loss.item()

        num_d_real_loss += 1

        # Backward (discriminator, real)

        d_real_loss.backward()



        # Forward (discriminator, fake; also generator forward pass)

        noise.normal_(0,1)

        target.fill_(0)

        noise_v = noise.detach()

        target_v = target.detach()

        g_output = g_net(noise_v)

        output = d_net(g_output.detach())

        # Compute loss (discriminator, fake)

        d_fake_loss = F.binary_cross_entropy(output, target_v)

        sum_d_fake_loss += d_fake_loss.item()

        num_d_fake_loss += 1

        # Backward (discriminator, fake)

        d_fake_loss.backward()



        # Update discriminator

        d_optimizer.step()



        # Reset generator gradients

        g_optimizer.zero_grad()



        # Don't compute gradients w.r.t. parameters for discriminator

        for p in d_net.parameters(): p.requires_grad = False

        # Forward (generator)

        target.fill_(1)

        target_v = target.detach()

        output = d_net(g_output)

        # Compute loss (generator)

        g_loss = F.binary_cross_entropy(output, target_v)

        sum_g_loss += g_loss.item()

        num_g_loss += 1

        # Backward (generator)

        g_loss.backward()



        # Update generator

        g_optimizer.step()



    # Checkpoint

    if epoch % save_every == 0:

        # Prepare generator state

        g_state_dict = g_net.state_dict()

        for k,v in g_state_dict.items():

            g_state_dict[k] = v.cpu()

        # Prepare generator state

        d_state_dict = d_net.state_dict()

        for k,v in d_state_dict.items():

            d_state_dict[k] = v.cpu()

        # Save checkpoint

        checkpoint_data = {"g_state_dict": g_state_dict, "d_state_dict": d_state_dict}

        torch.save(checkpoint_data, f"checkpoint-{epoch}.pth")

        

    # Show last generated batch

    g_image = torchvision.utils.make_grid(g_output.detach().cpu(), normalize=True).permute(1,2,0).numpy()

    figure(num=None, figsize=(15, 15))

    plt.axis("off")

    plt.imshow(g_image)

    plt.show()

    # Print

    avg_d_real_loss = sum_d_real_loss/num_d_real_loss

    avg_d_fake_loss = sum_d_fake_loss/num_d_fake_loss

    avg_g_loss = sum_g_loss/num_g_loss

    print(f"Epoch {epoch}: DR={avg_d_real_loss:.4f}, DF={avg_d_fake_loss:.4f}, G={avg_g_loss:.4f}")