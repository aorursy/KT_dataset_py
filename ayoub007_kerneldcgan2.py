import random



import torch

import torch.nn as nn

import torch.nn.parallel

import torch.optim as optim

from torch.utils.data import DataLoader



import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML



# set random seed

manual_seed = 999

random.seed(manual_seed)

torch.manual_seed(manual_seed)



from IPython import display

import time

import plotly

import plotly.io as pio



import plotly.graph_objects as go
from shutil import copyfile

copyfile(src = "../usr/lib/loggerscript/loggerscript.py", dst = "../working/utils.py")



from utils import Logger
batch_size = 128

image_size = 64

channels = 3

z_dim = 100

ngpu = 1



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
trans = transforms.Compose([

    transforms.Resize(image_size),

    transforms.CenterCrop(image_size),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
train_set = dset.ImageFolder(root='../input/celeba-aligned', transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)



print(len(train_set))
# Decide which device we want to run on

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



# Plot some training images

real_batch = next(iter(train_loader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
def weight_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):

    def __init__(self, leaky_relu=0, drop_out=0, n_layers=2):

        super(Generator, self).__init__()

        

        def convT_layer(in_size, out_size, kernel, stride, padding, LR= leaky_relu):

            return [

                nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel, stride=stride, padding=padding, bias=False),

                nn.LeakyReLU(negative_slope=LR, inplace=True),

                nn.Dropout2d(drop_out),

                nn.BatchNorm2d(out_size)]

        

        def out_layer(k=4):

            return [

                nn.ConvTranspose2d(64, 3, kernel_size=k, stride=2, padding=1, bias=False),

                nn.Tanh(),

            ]

        

        if(n_layers==2):

            self.network = nn.Sequential(

                #input layer:

                *convT_layer(z_dim, 512, 4, 1, 0), #512 (4*4)

                #2 hidden layers:

                *convT_layer(512, 128, 8, 2, 1), #128 (12*12) 

                *convT_layer(128, 64, 12, 2, 1),  #64 (32*32)

                #output layer:

                *out_layer() #3 (64*64)

                )

            

        if(n_layers==3):

            self.network = nn.Sequential(

                #input

                *convT_layer(z_dim, 512,4, 1, 0),

                #3 hidden layers

                *convT_layer(512, 256, 4, 2, 1),

                *convT_layer(256, 128, 4, 2, 1),

                *convT_layer(128, 64, 4, 2, 1),

                #output

                *out_layer(),

                )

            

        if(n_layers==4):

            self.network = nn.Sequential(

                nn.ConvTranspose2d(z_dim, 1024, 3, 1, 0), #1028 (3*3)



                *convT_layer(1024, 512, 3, 2, 1), #512 (5*5)

                *convT_layer(512, 256, 2, 2, 1), #256 (8*8)

                *convT_layer(256, 128, 2, 2, 1), #128 (14*14)

                *convT_layer(128, 64, 9, 2, 1), #64 (33*33)

                

                *out_layer(k=2), #3 (64*64)

            )

    

    def forward(self, x):

        return self.network(x)

# create the generator

netG = Generator(leaky_relu=0.2, n_layers=3).to(device)





# apply the weight_init function to randomly initialize all the weights

netG.apply(weight_init)



# print the model

print(netG)
tmp_noise = torch.randn(1, z_dim, 1, 1, device=device)

print(tmp_noise.shape)
class Discriminator(nn.Module):

    def __init__(self, leaky_relu=0, drop_out=0, n_layers=2):

        super(Discriminator, self).__init__()

        def conv_layer(in_size, out_size, kernel, stride, padding, DO=drop_out, LR= leaky_relu, bn=True):

            block = [

                nn.Conv2d(in_size, out_size, kernel_size=kernel, stride=stride, padding=padding, bias=False),

                nn.LeakyReLU(negative_slope=LR, inplace=True),

                nn.Dropout2d(DO)]

            if(bn): block.append(nn.BatchNorm2d(out_size))

            return block

        

        def out_layer(in_size, out_size, k=4):

            return [

                nn.Conv2d(in_size, out_size, kernel_size=k, stride=1, padding=0, bias=False),

                nn.Sigmoid()

            ]

        

        if(n_layers==2):

            self.network = nn.Sequential(

                *conv_layer(3, 64, 4, 2, 1, bn=False),

                *conv_layer(64, 256, 4, 2, 1),

                *conv_layer(256, 512, 8, 2, 1),

            )

            self.out = nn.Sequential(*out_layer(512, 1, 6))

            

        if(n_layers==3):

            self.network = nn.Sequential(

                *conv_layer(3, 64, 4, 2, 1, bn=False),

                *conv_layer(64, 128, 4, 2, 1),

                *conv_layer(128, 256, 4, 2, 1),

                *conv_layer(256, 512, 4, 2, 1),

                )

            self.out = nn.Sequential(*out_layer(512, 1))

            

        if(n_layers==4):

            self.network = nn.Sequential(

                *conv_layer(3, 64, 4, 2, 1, bn=False),

                *conv_layer(64, 128, 4, 2, 1),

                *conv_layer(128, 256, 4, 2, 1),

                *conv_layer(256, 512, 3, 2, 1),

                *conv_layer(512, 1028, 3, 2, 1),

            )

            self.out = nn.Sequential(*out_layer(1028, 1, 2))

        

    

    def forward(self, x):

        x = self.network(x)

        #if(matching):

            #return x

        return self.out(x)

netD = Discriminator(leaky_relu=0.2, drop_out=0.3, n_layers=3).to(device)



# initializing the weights

netD.apply(weight_init)



print(netD)
test2 = torch.randn(1, 3, 64, 64, device=device)

val = netD(test2)

print(test2.shape)

print(val)

print(val.shape)
# inititalize the BCELoss function

criterion = nn.BCELoss()



# Create batch of latent vectors that we will use to visualize

# the progression of the generator

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)



# Establish convention for real and fake labels during training

real_labels = 1

fake_labels = 0



# Setup Adam optimizers for both G and D

lr = 0.0002



g_optim = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

d_optim = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

print_every = 40

save_img_every = 500





test_samples = 16



iter_D_loss = []

iter_G_loss = []

D_loss = []

G_loss = []

iteration = 0
logger = Logger(model_name='DCGAN', data_name='CelebaFaces')

# training loop

print('starting training...')



for epoch in range(4):

    for i, data in enumerate(train_loader):

        

        ################################################################

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))  #

        ################################################################

        

        # train with all-real batch

        netD.zero_grad()

        # Format batch

        real = data[0].to(device)

        b_size = real.size(0)

        label = torch.full((b_size,), real_labels, device=device)

        

        # forward pass real batch through D

        output_real = netD(real).view(-1)

        

        # calculate loss on all-real batch

        d_loss_real = criterion(output_real, label)

        

        # calculate gradients for D in backward pass

        d_loss_real.backward()

        d_x = output_real.mean().item()

        

        ## Train with all-fake batch

        # Generate batch of latent vectors

        noise = torch.randn(b_size, z_dim, 1, 1, device=device)

        # generate fake images with G

        fake = netG(noise)

        label.fill_(fake_labels)

        # classify all fake batch with D

        output_fake = netD(fake.detach()).view(-1)

        # calculate D's loss on the all-fake batch

        d_loss_fake = criterion(output_fake, label)

        # Calculate the gradients for this batch

        d_loss_fake.backward()

        d_g_z1 = output_fake.mean().item()

        # add the gradients from the all-real and all-fake batches

        d_loss = d_loss_fake + d_loss_real

        # update D

        d_optim.step()

        

        ################################################

        # (2) Update G network: maximize log(D(G(z)))  #

        ################################################

        

        netG.zero_grad()

        label.fill_(real_labels)  # fake labels are real for generator cost

        

        # Since we just updated D, perform another forward pass of all-fake batch through D

        output = netD(fake).view(-1)

        

        # Calculate G's loss based on this output

        g_loss = criterion(output, label)

        # Calculate gradients for G

        g_loss.backward()

        d_g_z2 = g_loss.mean().item()

        # Update G

        g_optim.step()

        

        logger.log(d_loss, g_loss, epoch, i, 50)

        iter_G_loss.append(g_loss.item())

        iter_D_loss.append(d_loss.item())

        # Display Progress

        if (i) % 40 == 0:

            display.clear_output(True)

            # Display Images

            noise = torch.randn(b_size, 100, 1, 1, device=device)

        # Generate fake image batch with G

            fake = netG(noise).data.cpu()

            logger.log_images(fake, test_samples, epoch, i, 50);

            # Display status Logs

            logger.display_status(

                epoch, 3, i, 1583,

                d_loss, g_loss, output_real, output_fake

            )

            print('D loss: ', d_loss)

            print('G loss: ', g_loss)

            G_loss.append(iter_G_loss)

            D_loss.append(iter_D_loss)

            iter_G_loss = []

            iter_D_loss = []

            time.sleep(3)



print('end of training...')

len(G_loss[5])
test = G_loss[11] + G_loss[12] + G_loss[13] + G_loss[14] + G_loss[15]

test
len(D_loss[11])
test2 = D_loss[11] + D_loss[12] + D_loss[13] + D_loss[14] + D_loss[15]

test2
tmp_noise = torch.randn(1, z_dim, 1, 1, device=device)

print(tmp_noise.shape)

generated_tmp_img = netG(tmp_noise)

print(generated_tmp_img.shape)



tmp = torch.zeros(3, 64, 64)

tmp = generated_tmp_img[0, :, :, :]

host = tmp.cpu()

img_show = host.detach().numpy()

img_show = np.transpose(img_show, (1,2,0))

plt.imshow(img_show)
device_cpu = torch.device('cpu')

netG_cpu = netG.to(device_cpu)

netD_cpu = netD.to(device_cpu)
torch.save(netG_cpu.state_dict(), 'DCG_CELEB_2HL_Generator_half_trained.pth')

torch.save(netD_cpu.state_dict(), 'DCG_CELEB_2HL_Discrimiator_half_trained.pth')
print('half trained !')
import pandas as pd

list_attr_celeba = pd.read_csv("../input/celeba-dataset/list_attr_celeba.csv")

list_bbox_celeba = pd.read_csv("../input/celeba-dataset/list_bbox_celeba.csv")

list_eval_partition = pd.read_csv("../input/celeba-dataset/list_eval_partition.csv")

list_landmarks_align_celeba = pd.read_csv("../input/celeba-dataset/list_landmarks_align_celeba.csv")