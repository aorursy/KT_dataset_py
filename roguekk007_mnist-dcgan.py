import torch 

import torch.nn as nn 

import torch.nn.functional as F

import matplotlib.pyplot as plt

plt.gray()

from torchvision.datasets import MNIST

from torchvision.transforms import *

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



import os
G = nn.Sequential(

            nn.ConvTranspose2d(100, 128*4, 4, 1, 0),

            nn.InstanceNorm2d(128*4),

            nn.LeakyReLU(.2),



            nn.ConvTranspose2d(128*4, 128*2, 4, 2, 1),

            nn.InstanceNorm2d(128*2),

            nn.LeakyReLU(.2),



            nn.ConvTranspose2d(128*2, 128, 4, 2, 1),

            nn.InstanceNorm2d(128),

            nn.LeakyReLU(.2),



            nn.ConvTranspose2d(128, 1, 4, 2, 1),

            nn.Tanh()).cuda()



D = nn.Sequential(



            nn.Conv2d(1, 128, 4, 2, 1),

            nn.InstanceNorm2d(128),

            nn.LeakyReLU(.2),



            nn.Conv2d(128, 128*2, 4, 2, 1),

            nn.InstanceNorm2d(128*2),

            nn.LeakyReLU(.2),



            nn.Conv2d(128*2, 128*4, 4, 2, 1),

            nn.InstanceNorm2d(128*4),

            nn.LeakyReLU(.2),



            nn.Conv2d(128*4, 1, 4, 1, 0),

            nn.Sigmoid()).cuda()
def generate_latent(batch_size=32):

    return torch.randn(batch_size, 100, 1, 1)



# Test models

D(G(generate_latent(32).cuda())).shape



# Optimizer

D_op = torch.optim.Adam(D.parameters(), lr=5e-4, betas=(.5, .99))

G_op = torch.optim.Adam(G.parameters(), lr=5e-4, betas=(.5, .99))
batch_size = 1024

num_epochs = 30



# Data

dataset = MNIST(root='data', transform=Compose([Resize(32), ToTensor(), Normalize((.5,), (.5,))]),\

               download=True)

loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True)
criterion = nn.BCELoss()
# Display variables

columns = 4

rows = 3
ones, zeros = torch.ones(batch_size).cuda(), torch.zeros(batch_size).cuda()

for epoch in range(num_epochs):

    tk0 = tqdm(loader)

    

    print('Running epoch:', epoch+1)

    with torch.no_grad():

        z = generate_latent(rows*columns).cuda()

        fake = G(z)

        fake_imgs = fake.squeeze(0).cpu().numpy() / 2 + .5

        fig = plt.figure()

        for i in range(1, columns*rows +1):

            img = fake_imgs[i-1, 0, :, :]

            fig.add_subplot(rows, columns, i)

            plt.imshow(img)

            plt.axis('off')

        plt.show()

        

    for true, useless_label in tk0:

        # Train D

        D_op.zero_grad()

        z = generate_latent(batch_size).cuda()

        with torch.no_grad():

            fake = G(z)

        pred_fake = D(fake).view(-1)

        pred_true = D(true.cuda()).view(-1)

        D_loss = .5*criterion(pred_true, ones) + .5*criterion(pred_fake, zeros)

        D_loss.backward()

        D_op.step()



        # Train G                                                  

        G_op.zero_grad()                 

        z = generate_latent(batch_size).cuda()

        fake = G(z)             

        pred_fake = D(fake).view(-1)

        G_loss = criterion(pred_fake, ones)

        G_loss.backward()

        G_op.step()

                                        

        tk0.set_postfix({'G':G_loss.item(), 'D':D_loss.item()})