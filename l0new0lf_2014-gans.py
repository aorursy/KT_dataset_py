import os, time

from tqdm import tqdm

import numpy as np

import math

import matplotlib.pyplot as plt

import cv2 as cv2



# datasets

from torch.utils.data import DataLoader

from torchvision import datasets



# pre/post-processing

import torchvision.transforms as transforms

from torchvision.utils import save_image



# models

import torch.nn as nn

import torch.nn.functional as F

import torch
torch.manual_seed(123)

torch.cuda.manual_seed(123)

np.random.seed(123)
dataset = DataLoader(

    datasets.MNIST(

        'data/', 

        train=True, 

        download=True,

        transform = \

           transforms.Compose([

                transforms.ToTensor(),

                transforms.Normalize(0, 1)

            ])),

    batch_size=64, 

    shuffle=True)
BATCH_SIZE    = None

NUM_CHANNELS  = None

IMG_WIDTH     = None

IMG_HEIGHT    = None



# loop through first batch 

# to get dims

for X_batch, y_batch in dataset:

    BATCH_SIZE, NUM_CHANNELS, IMG_WIDTH, IMG_HEIGHT = X_batch.shape

    break
s = f"""

BATCH_SIZE    = {BATCH_SIZE}

NUM_CHANNELS  = {NUM_CHANNELS}

IMG_WIDTH     = {IMG_WIDTH}

IMG_HEIGHT    = {IMG_HEIGHT}

"""

print(s)
SINGLE_IMAGE_SHAPE = (NUM_CHANNELS, IMG_WIDTH, IMG_HEIGHT)
# Generator:

# ================================================================

# (num_smaples, 1, 10, 10) ~ P_z >>> (num_smaples, 1, 28, 28) ~ P_x



class Generator(nn.Module):

    def __init__(self, single_img_shape, z_w=10, z_h=10):

        super(Generator, self).__init__()

        self.single_img_shape = single_img_shape

        self.fc1 = nn.Linear(z_w*z_h, 128)

        self.fc2 = nn.Linear(128,512)

        self.fc3 = nn.Linear(512,1024 )

        self.fc4 = nn.Linear(1024,28*28)

        self.in1 = nn.BatchNorm1d(128)

        self.in2 = nn.BatchNorm1d(512)

        self.in3 = nn.BatchNorm1d(1024)



    def forward(self, x):

        x = F.leaky_relu(self.fc1(x),0.2)

        x = F.leaky_relu(self.in2(self.fc2(x)),0.2)

        x = F.leaky_relu(self.in3(self.fc3(x)),0.2)

        x = torch.tanh(self.fc4(x)) 

        return x.view(x.shape[0],*self.single_img_shape) # efficient reshape
# Discriminator:

# ================================================================

# (num_smaples, 1, 28, 28) ~ P_x >>> (num_samples, pred_probabs)



class Discriminator(nn.Module):

    def __init__(self, input_nodes):

        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_nodes, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128,1)



    def forward(self, x):

        x = x.view(x.size(0),-1)

        x = F.leaky_relu( self.fc1(x),0.2)

        x = F.leaky_relu(self.fc2(x),0.2)

        x = F.leaky_relu(self.fc3(x),0.2)

        x = torch.sigmoid(self.fc4(x))

        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = torch.nn.BCELoss()
"""

for epoch in range(0, 21):

    for batch_idx, (real_imgs, _) in enumerate(tqdm(dataset)):



        # avoid some stupid error in batch-processing :(

        if real_imgs.shape[0] != BATCH_SIZE:

            # print("==============================================")

            # print('+ Some runtime-error in batch size (skipping!)')

            # print("==============================================")

            break



        # real imgs: `real_imgs`

        # fake imgs: `fake_imgs` (created w/ updated W_g and real-time guassian noise)

        real_imgs = real_imgs.to(device)

        noise = torch.rand(BATCH_SIZE, 10*10).to(device)

        fake_imgs = generator(noise).to(device)



        # ground truths (not using mnist ys!)

        y_reals  = torch.ones((BATCH_SIZE, 1)).to(device)

        y_fakes = torch.zeros((BATCH_SIZE, 1)).to(device)





        # shuffle real and fake imgs data for better gradient

        # (remove temporality?). Consequently actual batch size becomes x2

        X_batch       = torch.cat((fake_imgs, real_imgs), 0).to(device)

        y_batch       = torch.cat((  y_fakes,   y_reals), 0).to(device)

        y_batch_inv   = torch.cat((  y_reals,   y_fakes), 0).to(device)

        shuffled_idxs = torch.randperm(BATCH_SIZE*2).to(device)



        # train discriminator

        # ===================================

        # maximize [neg. BCE] => minimize BCE

        # ===================================

        optimizer_D.zero_grad()

        d_loss = loss_func(discriminator(X_batch[shuffled_idxs]), y_batch[shuffled_idxs])

        d_loss.backward(retain_graph=True)

        optimizer_D.step()



        # train generator

        # ===================================

        # minimize [neg. BCE] => MAXIMIZE BCE

        # ===================================

        # Note: trick to maximize is to give exactly oppt. 

        # ground truths (only applicable for 2 classes)

        optimizer_G.zero_grad()

        g_loss = loss_func(discriminator(X_batch[shuffled_idxs]), y_batch_inv[shuffled_idxs])

        g_loss.backward()

        optimizer_G.step()



    # end of epoch

    print(f"epoch: {epoch} \tD Loss: {d_loss.item()}, \tG Loss: {g_loss.item()}")

    save_image(fake_imgs.data[:25], f'output/{epoch}_{batch_idx}.png', nrow=5, normalize=True)

"""
"""

import matplotlib.image as mpimg

def get_image(file_name):

    path = os.getcwd() + "/output/" + file_name + ".png"

    image = mpimg.imread(path)

    return image



fig, axarr = plt.subplots(1, 5, figsize=(20, 4))



for plot_idx, epoch in zip(range(0,5), range(0,21,5)):

    axarr[plot_idx].set_title(f"epoch: {epoch}")

    axarr[plot_idx].imshow(get_image(f"{epoch}_937"))



plt.show()

"""
start_feeding_fake_imgs_to_D = False

start_feeding_fake_imgs_to_D_e_num = 5



history = {}

for K in range(0, 100, 10):

    

    # avoid K=0

    if K == 0: K=1

    # configs some things (K-Tuning) Do the same for

    # Learning rate and other hyperparams

    os.makedirs(f'output_K{K}', exist_ok=True)

    history[f'{K}'] = {

        'G_loss': [],

        'D_loss': []

    }

    

    # ////////////////////////////////////////////////////////

    generator = Generator(SINGLE_IMAGE_SHAPE)

    discriminator = Discriminator(NUM_CHANNELS*IMG_WIDTH*IMG_HEIGHT)

    

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002,betas=(0.4,0.999))

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,betas=(0.4,0.999))

    

    generator.to(device)

    discriminator.to(device)

    loss_func.to(device)

    # ////////////////////////////////////////////////////////

    

    for epoch in range(0, 40):

        # stronger grads @early stages

        if epoch > start_feeding_fake_imgs_to_D_e_num: 

            start_feeding_fake_imgs_to_D = True



        for batch_idx, (real_imgs, _) in enumerate(tqdm(dataset)):



            # avoid some stupid error in batch-processing :(

            if real_imgs.shape[0] != BATCH_SIZE:

                # print("==============================================")

                # print('+ Some runtime-error in batch size (skipping!)')

                # print("==============================================")

                break



            # real imgs: `real_imgs`

            # fake imgs: `fake_imgs` (created w/ updated W_g and real-time guassian noise)

            real_imgs = real_imgs.to(device)

            noise = torch.rand(BATCH_SIZE, 10*10).to(device)

            fake_imgs = generator(noise).to(device)



            # ground truths (not using mnist ys!)

            y_reals  = torch.ones((BATCH_SIZE, 1)).to(device)

            y_fakes = torch.zeros((BATCH_SIZE, 1)).to(device)





            # shuffle real and fake imgs data for better gradient

            # (remove temporality?). Consequently actual batch size becomes x2

            X_batch       = torch.cat((fake_imgs, real_imgs), 0).to(device)

            y_batch       = torch.cat((  y_fakes,   y_reals), 0).to(device)

            y_batch_inv   = torch.cat((  y_reals,   y_fakes), 0).to(device)

            shuffled_idxs = torch.randperm(BATCH_SIZE*2).to(device)



            # train discriminator

            # ===================================

            # maximize [neg. BCE] => minimize BCE

            # ===================================

            if start_feeding_fake_imgs_to_D:

                # @later stages, feed REAL as well as FAKE images

                optimizer_D.zero_grad()

                d_loss = loss_func(discriminator(X_batch[shuffled_idxs]), y_batch[shuffled_idxs])

                d_loss.backward(retain_graph=True)

                optimizer_D.step()

            else:

                # @early stages, feed only REAL images

                # For stronger gradients initially

                optimizer_D.zero_grad()

                d_loss = loss_func(discriminator(real_imgs), y_reals)

                d_loss.backward(retain_graph=True)

                optimizer_D.step()



            # ////////////////////////////////////////////////////////////////

            # ////////////////////////////////////////////////////////////////

            # SLOW TRAINING of G (exactly as Algorithm in paper)

            # 1 W_g upadate for k W_d updates

            if batch_idx%K == 0:

                # train generator

                # ===================================

                # minimize [neg. BCE] => maximize BCE

                # ===================================

                # Note: trick to maximize is to give exactly oppt. 

                # ground truths (only applicable for 2 classes)

                optimizer_G.zero_grad()

                #g_loss = loss_func(discriminator(X_batch[shuffled_idxs]), y_batch_inv[shuffled_idxs])

                g_loss = loss_func(discriminator(fake_imgs), y_reals) # only fakes (w/ inverted labels)

                g_loss.backward()

                optimizer_G.step()

            # /////////////////////////////////////////////////////////////////

            # /////////////////////////////////////////////////////////////////



        # end of epoch

        print(f"K: {K}\t epoch: {epoch} \tD Loss: {d_loss.item()}, \tG Loss: {g_loss.item()}")

        history[f'{K}']['G_loss'].append(g_loss.item())

        history[f'{K}']['D_loss'].append(d_loss.item())

        save_image(fake_imgs.data[:25], f'output_K{K}/{epoch}_{batch_idx}.png', nrow=5, normalize=True)

        time.sleep(2)
import matplotlib.image as mpimg

def get_image(file_name, K):

    path = os.getcwd() + f"/output_K{K}/" + file_name + ".png"

    image = mpimg.imread(path)

    return image
plt.close()

fig, axarr = plt.subplots(2, 5, figsize=(20, 6))

axarr = axarr.reshape(-1)



for k_id, K in enumerate(range(0, 100, 10)):

    # avoid K=0

    if K == 0: K=1

    

    axarr[k_id].plot(

        list(range(0, len(history[f'{K}']['G_loss']))),

        history[f'{K}']['G_loss'],

        label = "G Loss"

    )

    

    axarr[k_id].plot(

        list(range(0, len(history[f'{K}']['D_loss']))),

        history[f'{K}']['D_loss'],

        label = "D Loss"

    )

    

    axarr[k_id].set_title(f"K: {K}")

    axarr[k_id].legend()



plt.show()
for K in range(0, 100, 10):

    # avoid K=0

    if K == 0: K=1

    

    for img_num in range(0, 40, 10):

        plt.close()

        fig, axarr = plt.subplots(1, 10, figsize=(20, 4))



        for plot_idx, epoch in zip(range(0,10), range(img_num,img_num+10,1)):

            axarr[plot_idx].set_title(f"K={K}\nepoch: {epoch}")

            axarr[plot_idx].imshow(get_image(f"{epoch}_937", K))



        plt.show()