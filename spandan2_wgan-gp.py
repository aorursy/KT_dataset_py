# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.chdir('..')

# Any results you write to the current directory are saved as output.
import numpy as np

import cv2

import matplotlib.pyplot as plt

import os

import torch

import torch.nn as nn

from torch.autograd import grad

from torch.autograd import Variable

import torchvision

import torchvision.transforms as transforms

import shutil

import statistics as st

import matplotlib.image as mpimg

import pylab



%pylab inline

pylab.rcParams['figure.figsize'] = (10, 10)
class generator(nn.Module):

    def __init__(self, dim_in, dim=64):

        super(generator,self).__init__()

        def genblock(dim_in, dim_out):

            block = nn.Sequential( nn.ConvTranspose2d(in_channels = dim_in, 

                                                      out_channels = dim_out,

                                                      kernel_size = 5, 

                                                      stride=2, 

                                                      padding=2,

                                                      output_padding = 1,

                                                      bias = False),

                                    nn.BatchNorm2d(dim_out),

                                    nn.ReLU()

                                    )

            return block

        def genimg(dim_in):

            block = nn.Sequential( nn.ConvTranspose2d(in_channels = dim_in, 

                                                      out_channels = 3,

                                                      kernel_size = 5, 

                                                      stride=2, 

                                                      padding=2,

                                                      output_padding = 1,

                                                      ),

                                    nn.Tanh()

                                    )

            return block

        

        self.prepare = nn.Sequential(nn.Linear(dim_in, dim*8*4*4, bias=False),

                                     nn.BatchNorm1d(dim*8*4*4),

                                     nn.ReLU())

        

        self.generate = nn.Sequential(genblock(dim*8, dim*4),

                                      genblock(dim*4, dim*2),

                                      genblock(dim*2, dim),

                                      genimg(dim))

    def forward(self, x):

        x = self.prepare(x)

        x = x.view(x.size(0), -1,4,4)

        x = self.generate(x)

        return x

#%%

class critic(nn.Module):

    def __init__(self, dim_in, dim=64):

        super(critic, self).__init__()

        

        def critic_block(dim_in , dim_out):

            block = nn.Sequential(nn.Conv2d(in_channels = dim_in, 

                                            out_channels = dim_out,

                                            kernel_size = 5, 

                                            stride=2, 

                                            padding=2),

                                    nn.InstanceNorm2d(dim_out, affine= True),

                                    nn.LeakyReLU(0.2))

            return block

        self.analyze = nn.Sequential(nn.Conv2d(in_channels = dim_in, 

                                               out_channels = dim,

                                               kernel_size = 5, 

                                               stride=2, 

                                               padding=2),

                                     nn.LeakyReLU(0.2),

                                     critic_block(dim,dim*2),

                                     critic_block(dim*2,dim*4),

                                     critic_block(dim*4, dim*8),

                                     nn.Conv2d(in_channels=dim*8, 

                                               out_channels=1,

                                               kernel_size=4))

    def forward(self,x):

        x = self.analyze(x)

        x =x.view(-1)

        return x
def gradient_penalty(x,y,f):

    shape =[x.size(0)] + [1] * (x.dim() -1)

    alpha = torch.rand(shape).cuda()

    z = x+ alpha *(y-x)

    z = Variable(z,requires_grad=True)

    z=z.cuda()

    o=f(z)

    g = grad(o,z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)

    gp = ((g.norm(p=2,dim=1))**2).mean()

    return gp

#%%

def save_checkpoint(state, save_path, is_best=False, max_keep=None):

    # save checkpoint

    torch.save(state, save_path)



    # deal with max_keep

    save_dir = os.path.dirname(save_path)

    list_path = os.path.join(save_dir, 'latest_checkpoint')



    save_path = os.path.basename(save_path)

    if os.path.exists(list_path):

        with open(list_path) as f:

            ckpt_list = f.readlines()

            ckpt_list = [save_path + '\n'] + ckpt_list

    else:

        ckpt_list = [save_path + '\n']



    if max_keep is not None:

        for ckpt in ckpt_list[max_keep:]:

            ckpt = os.path.join(save_dir, ckpt[:-1])

            if os.path.exists(ckpt):

                os.remove(ckpt)

        ckpt_list[max_keep:] = []



    with open(list_path, 'w') as f:

        f.writelines(ckpt_list)



    # copy best

    if is_best:

        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

#%%

def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):

    if os.path.isdir(ckpt_dir_or_file):

        if load_best:

            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')

        else:

            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:

                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])

    else:

        ckpt_path = ckpt_dir_or_file

    ckpt = torch.load(ckpt_path, map_location=map_location)

    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)

    return ckpt
epochs = 100 #Ideally go for 500+ epochs . Less epochs have been taken just for the purpose of this notebook

batch_size = 180

n_critic=5

lr=0.0002

z_dim = 100

transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

data = torchvision.datasets.ImageFolder('./input', transform = transform)

dataloader = torch.utils.data.DataLoader(data,

                                         batch_size=batch_size,

                                         shuffle=True,

                                         num_workers=3)

C = critic(3)

G = generator(z_dim)

C = C.cuda()

G = G.cuda()

print(C)

print(G)

print("Generator : ")

print(G)

print("Critic")

print(C)

start_epoch=0

G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))

C_opt = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.5,0.999))

#%%

checkpoint = './checkpoints/wgan_gp'

save_dir = './sample_images/wgan_gp'

if not isinstance(checkpoint, (list, tuple)):

    paths = [checkpoint]

    for path in paths:

        if not os.path.isdir(path):

            os.makedirs(path)

if not isinstance(save_dir, (list, tuple)):

    paths = [save_dir]

    for path in paths:

        if not os.path.isdir(path):

            os.makedirs(path)

try:

    ckpt = load_checkpoint(checkpoint)

    start_epoch = ckpt['epoch']

    C.load_state_dict(ckpt['D'])

    G.load_state_dict(ckpt['G'])

    C_opt.load_state_dict(ckpt['d_optimizer'])

    G_opt.load_state_dict(ckpt['g_optimizer'])

except:

    print(' [*] No checkpoint!')

    start_epoch = 0

#%%

z_sample = Variable(torch.randn(100, z_dim)).cuda()

import time

for epoch in range(start_epoch, epochs):

    start_time = time.time()

    C_loss= []

    G_loss=[]

    G.train()

    for i, (images, _) in enumerate(dataloader):

        step = epoch * len(dataloader) + i + 1



        images = Variable(images)

        batch = images.size(0)

        images = images.cuda()

        z = Variable(torch.randn(batch, z_dim))

        z = z.cuda()

        

        generated = G(z)

        real_criticized = C(images)

        fake_criticized = C(generated)

        

        em_distance = real_criticized.mean() - fake_criticized.mean()

        grad_penalty = gradient_penalty(images.data, generated.data, C)

        

        CriticLoss = -em_distance + grad_penalty*10

        C_loss.append(CriticLoss.item())

        C.zero_grad()

        CriticLoss.backward()

        C_opt.step()

        

        if step % n_critic == 0:

            z = Variable(torch.randn(batch, z_dim))

            z = z.cuda()

            generated = G(z)

            fake_criticized = C(generated)

            GenLoss = -fake_criticized.mean()

            G_loss.append(GenLoss.item())

            C.zero_grad()

            G.zero_grad()

            GenLoss.backward()

            G_opt.step()

            print("Epoch {} : {}/{} :: {} mins".format(epoch+1, i+1, len(dataloader), (time.time()-start_time)/60), end='\r')         

    print("Epoch {} completed".format(epoch+1))

    G.eval()

    fake_gen_images = (G(z_sample).data +1)/2.0

    torchvision.utils.save_image(fake_gen_images, save_dir+'/Epoch '+str(epoch+1)+".jpg",nrow=10)

    save_checkpoint({'epoch': epoch + 1,

                           'D': C.state_dict(),

                           'G': G.state_dict(),

                           'd_optimizer': C_opt.state_dict(),

                           'g_optimizer': G_opt.state_dict()},

                          '%s/Epoch_(%d).ckpt' % (checkpoint, epoch + 1),

                          max_keep=2)
a= os.listdir('sample_images/wgan_gp/')

a.sort()

print(a[-1])

img=mpimg.imread('sample_images/wgan_gp/'+a[-1])

plt.imshow(img)

plt.axis('off')

plt.show()
!wget https://raw.githubusercontent.com/spandan2/Wgan-GP_cats/master/sample_images/wgan_gp/Epoch%20500.jpg

img=mpimg.imread('Epoch 500.jpg')

plt.imshow(img)

plt.axis('off')

plt.show()