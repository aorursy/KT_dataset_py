# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import xml.etree.ElementTree as ET



import torch

import torchvision



# for testing only

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

# This loader will use the underlying loader plus crop the image based on the annotation

def doggo_loader(path):

    img = torchvision.datasets.folder.default_loader(path) # default loader

    

    # Get bounding box

    annotation_basename = os.path.splitext(os.path.basename(path))[0]

    annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))

    annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)

    tree = ET.parse(annotation_filename)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox')

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

    bbox = (xmin, ymin, xmax, ymax)

    

    # return cropped image

    return img.crop(bbox)





# The dataset (example)

dataset = torchvision.datasets.ImageFolder(

    '../input/all-dogs/',

    loader=doggo_loader, # THE CUSTOM LOADER

    transform=torchvision.transforms.Compose([

        torchvision.transforms.Resize(70),

        torchvision.transforms.CenterCrop(64),

        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]) # some transformations, add your data preprocessing here

)
# Check that it all loads without a bug

for i in tqdm(range(len(dataset))):

    _ = dataset[i]

print('Ok.')
# Check that we get only the CUTE DOGS OH YES WHOS THE GOOD DOGGO ITS YOU

n = 10

_, axes = plt.subplots(figsize=(4*n, 4*n), ncols=n, nrows=n)

for i, ax in enumerate(axes.flatten()):

    ax.imshow(dataset[i][0].permute(1, 2, 0).detach().numpy())

plt.show()

len(dataset)

train_loader = torch.utils.data.DataLoader(dataset=dataset,

                                          batch_size=128,

                                          shuffle=True,

                                          num_workers=0)
# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

plot_size=20

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    ax.imshow(np.transpose(images[idx], (1, 2, 0)))

    # print out the correct label for each image

    # .item() gets the value contained in a Tensor

    ax.set_title(str(labels[idx].item()))
# current range

img = images[0]



print('Min: ', img.min())

print('Max: ', img.max())
# # helper scale function

# def scale(x, feature_range=(-1, 1)):

#     ''' Scale takes in an image x and returns that image, scaled

#        with a feature_range of pixel values from -1 to 1. 

#        This function assumes that the input x is already scaled from 0-1.'''

#     # assume x is scaled to (0, 1)

#     # scale to feature_range and return scaled x

#     min, max = feature_range

#     x = x * (max - min) + min

#     return x

# # scaled range

# scaled_img = scale(img)



# print('Scaled min: ', scaled_img.min())

# print('Scaled max: ', scaled_img.max())
import torch.nn as nn

import torch.nn.functional as F



# helper conv function

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    """Creates a convolutional layer, with optional batch normalization.

    """

    layers = []

    conv_layer = nn.Conv2d(in_channels, out_channels, 

                           kernel_size, stride, padding, bias=False)

    

    # append conv layer

    layers.append(conv_layer)



    if batch_norm:

        # append batchnorm layer

        layers.append(nn.BatchNorm2d(out_channels))

     

    # using Sequential container

    return nn.Sequential(*layers)

class Discriminator(nn.Module):



    def __init__(self, conv_dim=64):

        super(Discriminator, self).__init__()



        # complete init function

        self.conv_dim = conv_dim



        # 64x64 input

        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # first layer, no batch_norm

        # 32x32 out

        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # 16x16 out

        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        # 8x8 out

        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)

        

        # final, fully-connected layer

        self.fc = nn.Linear(conv_dim*8*4*4, 1)



    def forward(self, x):

        # all hidden layers + leaky relu activation

        out = F.leaky_relu(self.conv1(x), 0.2)

        out = F.leaky_relu(self.conv2(out), 0.2)

        out = F.leaky_relu(self.conv3(out), 0.2)

        out = F.leaky_relu(self.conv4(out), 0.2)

        

        # flatten

        out = out.view(-1, self.conv_dim*8*4*4)

        out = self.fc(out)        

        return out   

                # final output layer

        

    
# helper deconv function

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    """Creates a transposed-convolutional layer, with optional batch normalization.

    """

    # create a sequence of transpose + optional batch norm layers

    layers = []

    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 

                                              kernel_size, stride, padding, bias=False)

    # append transpose convolutional layer

    layers.append(transpose_conv_layer)

    

    if batch_norm:

        # append batchnorm layer

        layers.append(nn.BatchNorm2d(out_channels))

        

    return nn.Sequential(*layers)

class Generator(nn.Module):

    

    def __init__(self, z_size, conv_dim=64):

        super(Generator, self).__init__()



        # complete init function

        

        self.conv_dim = conv_dim

        

        # first, fully-connected layer

        self.fc = nn.Linear(z_size, conv_dim*8*4*4)



        # transpose conv layers

        self.t_conv1 = deconv(conv_dim*8, conv_dim*4, 4)

        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)

        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4)

        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)

        



    def forward(self, x):

        # fully-connected + reshape 

        out = self.fc(x)

        out = out.view(-1, self.conv_dim*8, 4, 4) # (batch_size, depth, 4, 4)

        

        # hidden transpose conv layers + relu

        out = F.relu(self.t_conv1(out))

        out = F.relu(self.t_conv2(out))

        out = F.relu(self.t_conv3(out))

        

        # last layer + tanh activation

        out = self.t_conv4(out)

        out = torch.tanh(out)

        

        return out

    
# define hyperparams

conv_dim = 64

z_size = 100



# define discriminator and generator

D = Discriminator(conv_dim)

G = Generator(z_size=z_size, conv_dim=conv_dim)



print(D)

print()

print(G)
train_on_gpu = torch.cuda.is_available()



if train_on_gpu:

    # move models to GPU

    G.cuda()

    D.cuda()

    print('GPU available for training. Models moved to GPU')

else:

    print('Training on CPU.')

    
def real_loss(D_out, smooth=False):

    batch_size = D_out.size(0)

    # label smoothing

    if smooth:

        # smooth, real labels = 0.9

        labels = torch.ones(batch_size)*0.9

    else:

        labels = torch.ones(batch_size) # real labels = 1

    # move labels to GPU if available     

    if train_on_gpu:

        labels = labels.cuda()

    # binary cross entropy with logits loss

    criterion = nn.BCEWithLogitsLoss()

    # calculate loss

    loss = criterion(D_out.squeeze(), labels)

    return loss



def fake_loss(D_out):

    batch_size = D_out.size(0)

    labels = torch.zeros(batch_size) # fake labels = 0

    if train_on_gpu:

        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()

    # calculate loss

    loss = criterion(D_out.squeeze(), labels)

    return loss
import torch.optim as optim



# params

lr = 0.0002

beta1=0.5

beta2=0.999 # default value



# Create optimizers for the discriminator and generator

d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])

g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
import pickle as pkl



# training hyperparams

num_epochs = 6



# keep track of loss and generated, "fake" samples

samples = []

losses = []



print_every = 300



# Get some fixed data for sampling. These are images that are held

# constant throughout training, and allow us to inspect the model's performance

sample_size=16

fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))

fixed_z = torch.from_numpy(fixed_z).float()



# train the network

for epoch in range(num_epochs):

    

    for batch_i, (real_images, _) in enumerate(train_loader):

                

        batch_size = real_images.size(0)

        

        # important rescaling step

        real_images = (real_images)

        

        # ============================================

        #            TRAIN THE DISCRIMINATOR

        # ============================================

        

        d_optimizer.zero_grad()

        

        # 1. Train with real images



        # Compute the discriminator losses on real images 

        if train_on_gpu:

            real_images = real_images.cuda()

        D_real = D(real_images)

        d_real_loss = real_loss(D_real)

        

        # 2. Train with fake images

        

        # Generate fake images

        z = np.random.uniform(-1, 1, size=(batch_size, z_size))

        z = torch.from_numpy(z).float()

        # move x to GPU, if available

        if train_on_gpu:

            z = z.cuda()

        fake_images = G(z)

        

        # Compute the discriminator losses on fake images            

        D_fake = D(fake_images)

        d_fake_loss = fake_loss(D_fake)

        

        # add up loss and perform backprop

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()

        d_optimizer.step()

        

        

        # =========================================

        #            TRAIN THE GENERATOR

        # =========================================

        g_optimizer.zero_grad()

        

        # 1. Train with fake images and flipped labels

        

        # Generate fake images

        z = np.random.uniform(-1, 1, size=(batch_size, z_size))

        z = torch.from_numpy(z).float()

        if train_on_gpu:

            z = z.cuda()

        fake_images = G(z)

        

        # Compute the discriminator losses on fake images 

        # using flipped labels!

        D_fake = D(fake_images)

        g_loss = real_loss(D_fake) # use real loss to flip labels

        

        # perform backprop

        g_loss.backward()

        g_optimizer.step()



        # Print some loss stats

        if batch_i % print_every == 0:

            # append discriminator loss and generator loss

            losses.append((d_loss.item(), g_loss.item()))

            # print discriminator and generator loss

            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(

                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))



    

    ## AFTER EACH EPOCH##    

    # generate and save sample, fake images

    G.eval() # for generating samples

    if train_on_gpu:

        fixed_z = fixed_z.cuda()

    samples_z = G(fixed_z)

    samples.append(samples_z)

    G.train() # back to training mode





# Save training generator samples

with open('train_samples.pkl', 'wb') as f:

    pkl.dump(samples, f)
fig, ax = plt.subplots()

losses = np.array(losses)

plt.plot(losses.T[0], label='Discriminator', alpha=0.5)

plt.plot(losses.T[1], label='Generator', alpha=0.5)

plt.title("Training Losses")

plt.legend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image

if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

sample_size=50

fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))

fixed_z = torch.from_numpy(fixed_z).float()

im_batch_size = 50

n_images=10000

for i_batch in range(0, n_images, im_batch_size):

    G.eval() # for generating samples

    if train_on_gpu:

        fixed_z = fixed_z.cuda()

    gen_images = G(fixed_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
import shutil

shutil.make_archive('images', 'zip', '../output_images')