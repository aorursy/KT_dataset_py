%reload_ext autoreload

%autoreload 2

%matplotlib inline
import torch

import torch.nn as nn

import torchvision

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.utils.data.sampler

import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader as DL

from torch.utils.data import *

from PIL import Image, ImageFilter

import os

import cv2

import numpy

import random

import fnmatch

import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
X_train = np.load('/kaggle/input/lsunchurch/church_outdoor_train_lmdb_color_64.npy')

print(X_train.shape)
bs=16
#X_train = X_train[0:(126227-(126227%bs))]

#X_train = X_train[0:16000]
X_test = X_train[16000:17600]

X_train = X_train[0:16000]
d = DataLoader(X_train, batch_size=bs)
test_loader = DataLoader(X_test, batch_size=bs)
len(d)

len(test_loader)
def en_double_conv(in_channels, out_channels):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),

        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),

        nn.BatchNorm2d(out_channels),

        nn.LeakyReLU(0.2, inplace=True)

    )



def dec_double_conv(in_channels, out_channels):

  return nn.Sequential(

      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),

      nn.ReLU(inplace=True),

      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),

      nn.BatchNorm2d(out_channels),

      nn.ReLU(inplace=True)

  )



n=6



class G(nn.Module):

    def __init__(self):

        super().__init__()



        self.dconv_1 = en_double_conv(3, n)

        self.dconv_2 = en_double_conv(n, n*2)

        self.dconv_3 = en_double_conv(n*2, n*4)

        self.dconv_4 = en_double_conv(n*4, n*8)

        self.dconv_5 = en_double_conv(n*8, n*8*2)

        self.dconv_6 = en_double_conv(n*8*2, n*8*3)



        self.dropout = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool2d(2)



        self.TConv6 = nn.ConvTranspose2d(n*8*3, n*8*3, 4, 2, 1)

        self.TConv5 = nn.ConvTranspose2d(n*8*5, n*8*5, 4, 2, 1)

        self.TConv4 = nn.ConvTranspose2d(n*8*6, n*8*6, 4, 2, 1)

        self.TConv3 = nn.ConvTranspose2d(n*52, n*52, 4, 2, 1)

        self.TConv2 = nn.ConvTranspose2d(n*54, n*54, 4, 2, 1)

        self.TConv1 = nn.ConvTranspose2d(n*55, 3, 4, 2, 1)

        

    def forward(self, x):

        conv1 = self.dconv_1(x)

        conv1 = self.maxpool(conv1)



        conv2 = self.dconv_2(conv1)

        conv2 = self.maxpool(conv2)



        conv3 = self.dconv_3(conv2)

        conv3 = self.maxpool(conv3)



        conv4 = self.dconv_4(conv3)

        conv4 = self.maxpool(conv4)



        conv5 = self.dconv_5(conv4)

        conv5 = self.maxpool(conv5)



        conv6 = self.dconv_6(conv5)

        conv6 = self.maxpool(conv6)



        x = self.TConv6(conv6)



        x = torch.cat([x, conv5], dim=1)

        x = self.TConv5(x)

        #x = self.dropout(x)



        x = torch.cat([x, conv4], dim=1)

        x = self.TConv4(x)

        #x = self.dropout(x)



        x = torch.cat([x, conv3], dim=1)

        x = self.TConv3(x)

        #x = self.dropout(x)



        x = torch.cat([x, conv2], dim=1)

        x = self.TConv2(x)

        #x = self.dropout(x)

        

        x = torch.cat([x, conv1], dim=1)

        x = self.TConv1(x)

        x = nn.Tanh()(x)



        return x
class D(nn.Module):

  def __init__(self):

    super(D, self).__init__()

    self.main = nn.Sequential(        

        nn.Conv2d(3, 64, 2, 2, 0),

        nn.LeakyReLU(0.2),

        

        nn.Conv2d(64, 128, 2, 2, 0),

        nn.BatchNorm2d(128),

        nn.LeakyReLU(0.2),

        

        nn.Conv2d(128, 256, 2, 2, 0),

        nn.BatchNorm2d(256),

        nn.LeakyReLU(0.2),

        

        nn.Conv2d(256, 512, 2, 2, 0),

        nn.BatchNorm2d(512),

        nn.LeakyReLU(0.2),

        

        nn.Conv2d(512, 1024, 2, 2, 0),

        nn.BatchNorm2d(1024),

        nn.LeakyReLU(0.2),

        

        nn.Conv2d(1024, 1, 2, 2, 0),

        nn.Sigmoid()

    )

    

  def forward(self, im):

    return self.main(im)
Generator = G().to(device)

Discriminator = D().to(device)
def weights_init(m):

    if isinstance(m, nn.Conv2d):

        m.weight.data.normal_(0, 0.02)

        m.bias.data.normal_(0, 0.001)
weights_init(Generator)

weights_init(Discriminator)
print("Number of parameters in Generator: ", sum([p.numel() for p in Generator.parameters()]))

print("Number of parameters in Discriminator: ", sum([p.numel() for p in Discriminator.parameters()]))
criterion = nn.BCELoss()

adv_criterion = nn.BCELoss()

l1_criterion = nn.L1Loss()

G_optim = torch.optim.Adam(Generator.parameters(), lr=1e-4)

D_optim = torch.optim.Adam(Discriminator.parameters(), lr=1e-4)
Discriminator.train()

Generator.train()
def save_pic(epoch_no, im):

  Generator.eval()

  im = im.unsqueeze(0).to(device)



  #transforms.ToPILImage()(im[0]).save("input.jpg") #save input image.



  output = Generator(im)

  

  output = output[0].detach().cpu()

  output = output.clamp(0.0, 1.0)



  PIL_img = transforms.ToPILImage()(output)

  PIL_img = PIL_img.save(str(epoch_no) + ".jpg")

  Generator.train()
D_losses_train = []

G_losses_train = []



D_losses_test = []

G_losses_test = []
def shuffle_data(fake_im, real_im):

  batch_size=fake_im.shape[0]

  data=torch.cat((fake_im, real_im),dim=0)

  labels=torch.cat((torch.zeros(batch_size), torch.ones(batch_size)))

  

  return data, labels
def augment_image(real_im): #crop nxn region of images, real_crop, return mask

  augmented_image = real_im

  

  #transforms.ToPILImage()(real_im[0]).save("pre-augmentation.jpg")



  a_size=16

  mask = torch.ones(real_im.shape)

  real_crop = torch.zeros(real_im.shape[0], 3, a_size, a_size)



  for i in range(real_im.shape[0]):

    lim = real_im.shape[2]-a_size



    #r_x = random.randrange(0, lim-1)

    #r_y = random.randrange(0, lim-1)



    r_x = 24

    r_y = 24



    for x in range(a_size):

      for y in range(a_size):

        for c in range(3):

          augmented_image[i][c][r_x+x][r_y+y]=0.5

          real_crop[i][c][x][y]=real_im[i][c][r_x+x][r_y+y]

          mask[i][c][r_x+x][r_y+y]=1000



  #transforms.ToPILImage()(augmented_image[0]).save("augmented.jpg")



  return augmented_image, real_crop, mask
n_epochs = 10



for epoch in range(n_epochs):

  D_train_loss = 0.0

  G_train_loss = 0.0



  D_test_loss = 0.0

  G_test_loss = 0.0



  for i, real_im in enumerate(d):

    real_im = real_im.permute(0, 3, 1, 2)

    real_im = real_im.type(torch.FloatTensor)/255



    augmented_im, real_crop, mask = augment_image(real_im)

    augmented_im = augmented_im.to(device)

    mask = mask.to(device)

    #real_crop = real_crop.to(device)

    real_im = real_im.type(torch.FloatTensor)

    

    if i%1599==0:

      save_pic(epoch, real_im[0])



    ##########Train the discriminator##########

    D_optim.zero_grad()

    real_im=real_im.to(device)

    fake_img = Generator(augmented_im)



    #transforms.ToPILImage()(real_im[0]).save("real_im.jpg")



    data, labels = shuffle_data(fake_img, real_im)

    guess = Discriminator(data)

    

    D_loss = criterion(guess, labels.to(device))

    D_train_loss += D_loss.item()

    D_loss.backward()

    D_optim.step()

    ###########################################

    

    ############Train the generator############ (curriculum training, Foreground-aware II)



    ##########L1 loss update############

    G_optim.zero_grad()

    fake_img = Generator(augmented_im)



    G_loss_l1 = l1_criterion(fake_img, real_im)

    G_train_loss += G_loss_l1.item() #for plotting loss

    G_loss_l1.backward(retain_graph=True)

    G_optim.step()

    ####################################

    

    ##########Adversarial loss update###

    G_optim.zero_grad()

    #fake_img = Generator(augmented_im) #no need for this

    guess = Discriminator(fake_img).view(-1)

    G_loss_adv = adv_criterion(guess, torch.ones(bs).to(device))*1e-2

    G_train_loss += G_loss_adv.item() #for plotting loss

    G_loss_adv.backward()

    G_optim.step()

    ####################################

    

    ###########################################

  

  for i, real_im in enumerate(test_loader):

    Generator.eval()

    Discriminator.eval()

    

    real_im = real_im.permute(0, 3, 1, 2)

    real_im = real_im.type(torch.FloatTensor)/255

    

    augmented_img, _, _ = augment_image(real_im)

    augmented_img = augmented_img.to(device)

    real_im = real_im.to(device)

    

    fake_img = Generator(augmented_img)

    guess = Discriminator(augmented_img)

    adv_loss = adv_criterion(guess, torch.ones(bs).to(device))

    l1_loss = l1_criterion(augmented_img, real_im)

    G_loss = adv_loss*1e-2 + l1_loss

    G_test_loss += G_loss.item()

    

    Generator.train()

    Discriminator.train()

    

  G_train_loss = G_train_loss/16000

  G_test_loss = G_test_loss/1600

  print("Epoch " + str(epoch) + ", Train: " + str(G_train_loss))# + " , Test: " + str(G_test_loss))

  G_losses_train.append(G_train_loss)

  #G_losses_test.append(G_test_loss)

  #save_pic(epoch)
torch.save(Generator.state_dict(), "lsunChurchGenerator")

torch.save(Discriminator.state_dict(), "lsunChurchDiscriminator")

torch.save(G_optim.state_dict(), "lsunChurchG_optim")

torch.save(D_optim.state_dict(), "lsunChurchD_optim")