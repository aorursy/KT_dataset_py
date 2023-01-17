import sys

print(sys.version)

device='cuda' #Changing Device to Run on GPU

data_path="/kaggle/input/celeba-dataset/img_align_celeba/"

from PIL import Image

import os

from skimage import io, transform

from skimage import io, transform



import random

import time

import itertools

import pandas as pd

import numpy as np



from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

from torch.autograd import Variable

from scipy import ndimage

from IPython.display import display

import torchvision.datasets as datasets

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import transforms



from torch.utils.data import Dataset, DataLoader





print(torch.version.cuda)  

print(torch.cuda.device_count())

print(torch.cuda.is_available())



transforms_=transforms.Compose([

                               transforms.Resize(64),

                               transforms.CenterCrop(64),

                               transforms.ToTensor(),

                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                           ])

dataset = datasets.ImageFolder(root=data_path,

                           transform=transforms_)

# Create the dataloader

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,

                                         shuffle=True)
import torchvision.utils as vutils

real_batch = next(iter(dataloader))

print(real_batch[0].size())

print(len(dataloader))


plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
Z_gen_size=100


class Generator(nn.Module):

  def __init__(self):

    super(Generator, self).__init__()

    self.fct1= nn.ConvTranspose2d(Z_gen_size,256,4,1,0) #4*4 out

    self.fct2= nn.ConvTranspose2d(256, 128, 4,2,1)  #8*8 out

    self.fct3= nn.ConvTranspose2d(128,64, 4, 2, 1)  #16*16

    self.fct4= nn.ConvTranspose2d(64,32, 4,2,1)  #32*32

    self.fct5= nn.ConvTranspose2d(32,3, 4,2,1)  #64x64

    

    self.norm1_2d= nn.BatchNorm2d(256)

    self.norm2_2d= nn.BatchNorm2d(128)

    self.norm3_2d=nn.BatchNorm2d(64)

    self.norm4_2d=nn.BatchNorm2d(32)

    

    



  def forward(self,x):



    #FC1

    x= self.fct1(x)

    x= self.norm1_2d(x)

    x= F.leaky_relu(x,0.2)

    #FC2

    x= self.fct2(x)    

    x= self.norm2_2d(x)

    x= F.leaky_relu(x,0.2)

    #FC3

    x= self.fct3(x)    

    x= self.norm3_2d(x)

    x= F.leaky_relu(x,0.2)

    

    #FC3

    x= self.fct4(x)    

    x= self.norm4_2d(x)

    x= F.leaky_relu(x,0.2)

    

   #FC3

    x= self.fct5(x)

    x= F.tanh(x)

      

    

    return x



genr=Generator()

genr=genr.float()

genr.to(device)
class Discriminator(nn.Module):

  def __init__(self):

    super(Discriminator, self).__init__()

    self.conv1 = nn.Conv2d(3,32,4,2,1)    #32x32

    self.conv2 = nn.Conv2d(32,64,4,2,1)    #16x16

    self.conv3=  nn.Conv2d(64,128,4,2,1)  #8*8

    self.conv4 = nn.Conv2d(128,256,4,2,1) #4*4

    self.conv5 = nn.Conv2d(256,1,4,1,0)   #1*1

    self.drop1 = nn.Dropout(0.3)

    self.norm1_2d=nn.BatchNorm2d(32)

    self.norm2_2d=nn.BatchNorm2d(64)

    self.norm3_2d=nn.BatchNorm2d(128)

    self.norm4_2d=nn.BatchNorm2d(256)

    



  def forward(self,x):



   

    #Three fully connected Layers

    

    #FC1

    x= self.conv1(x)

    x= self.norm1_2d(x)

    x= F.leaky_relu(x,0.2)    

    

    #FC2

    x= self.conv2(x)    

    x= self.norm2_2d(x)

    x= F.leaky_relu(x,0.2)

    x= self.drop1(x)



    #Fc3

    x= self.conv3(x)

    x= self.norm3_2d(x)

    x= F.leaky_relu(x,0.2)

       



    #FC4

    x= self.conv4(x)

    x= self.norm4_2d(x)

    x= F.leaky_relu(x,0.2)

    

    

    x= self.conv5(x)

    x=x.view(-1)

    #x= torch.sigmoid(x)

    





    return x



discr=Discriminator()

discr=discr.float()

discr.to(device)
def dis_loss_fn(data_outs, gen_outs, smooth=1):

  targets_d= torch.ones(data_outs.size()[0], dtype=torch.float64, device=device)*smooth

  targets_g= torch.zeros(gen_outs.size()[0], dtype=torch.float64, device=device)

  loss= nn.BCEWithLogitsLoss()

  loss_calc=loss(data_outs, targets_d) + loss(gen_outs, targets_g)

  return loss_calc



def gen_loss_fn(gen_outs):

  targets= torch.ones(gen_outs.size()[0], dtype=torch.float64, device=device)

  loss= nn.BCEWithLogitsLoss()

  loss_calc=loss(gen_outs, targets)

  return loss_calc
z_e=np.random.uniform(-1, 1, size=(1, Z_gen_size,1,1))

z_e=torch.from_numpy(z_e).float().to(device)
dis_optim=optim.Adam(discr.parameters(), lr= 0.0002)

gen_optim=optim.Adam(genr.parameters(), lr= 0.0002)
epochs=40 #No. Of Epochs

k=1

batch_loss_dis=[] #for storing losses of individual batches while training

batch_loss_gen=[] 



total_epochs=len(dataloader) 

for st in range(0,epochs):



  

  for i, (x_batch, y_batch) in tqdm(enumerate(dataloader),position=0, leave=True, total=total_epochs):

    

    x_batch=x_batch.to(device)

    z_batch=np.random.uniform(-1, 1, size=(x_batch.size()[0], Z_gen_size,1,1))

    z_batch=torch.from_numpy(z_batch).float().to(device)



    discr.train()

    genr.train()

    for q in range(0,k):

      dis_optim.zero_grad()

      dis_outs =discr(x_batch)

     



      gen_img  = genr(z_batch)

      gen_outs = discr(gen_img)

     

      dis_loss= dis_loss_fn(data_outs= dis_outs, gen_outs= gen_outs, smooth=0.9)

      dis_loss.backward()

      dis_optim.step()

      batch_loss_dis.append(dis_loss.item())

    

    gen_optim.zero_grad()

    z_batch=np.random.uniform(-1, 1, size=(x_batch.size()[0], Z_gen_size,1,1))

    z_batch=torch.from_numpy(z_batch).float().to(device)

    gen_img  = genr(z_batch)    

    gen_outs = discr(gen_img)



    gen_loss= gen_loss_fn(gen_outs)

    gen_loss.backward()

    gen_optim.step()

        

    batch_loss_gen.append(gen_loss.item())





  #Tests And Eval

  dis_loss_ep=sum(batch_loss_dis)/len(batch_loss_dis)

  gen_loss_ep=sum(batch_loss_gen)/len(batch_loss_gen)

  genr.eval()

  with torch.no_grad():

    example=genr(z_e)

    example =example.cpu().numpy().reshape(3,64,64)

    print(example.shape)

    plt.imshow(example.transpose(1,2,0)*0.5 +0.5)

    plt.show()

 

 

  print("Epoch", str(st+1)+"/" + str(epochs))

  print("DIS_Loss:", dis_loss_ep,"      ", "GEN_Loss:",  gen_loss_ep )

  
genr.eval()

with torch.no_grad():

    example=genr(z_e)

    example =example.cpu().numpy().reshape(3,64,64)

    print(example.shape)

    plt.imshow(example.transpose(1,2,0)*0.5 +0.5)

    plt.show()

plt.figure(figsize=(10,5))

plt.title("Generator and Discriminator Loss During Training", color='yellow')

plt.plot(batch_loss_gen,label="G")

plt.plot(batch_loss_dis,label="D", color='red')

plt.xlabel("iterations")

plt.ylabel("Loss")

plt.legend()

plt.show()
z_examples=np.random.uniform(-1, 1, size=(256, Z_gen_size,1,1))

z_examples=torch.from_numpy(z_examples).float().to(device)
genr.eval()



import torchvision.utils as vutils



with torch.no_grad():

  example=genr(z_examples)

  example_f =example.detach().cpu()



grid_img=vutils.make_grid(example_f,16, padding=2 , normalize=True).numpy().transpose(1,2,0)

print("Generated Images")



img = Image.fromarray( np.uint8((grid_img)*256),'RGB')

display(img)
example_nm =example_f.cpu().numpy()



plt.figure(figsize=(16,16)) 



for i in range(256):

    plt.subplot(16,16,i+1) 

    plt.axis("off")   

    plt.imshow(example_nm[i].transpose(1,2,0)*0.5 +0.5)

plt.show()