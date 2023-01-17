from distutils.dir_util import copy_tree

import os
from_path = "../input/notebook_ims/notebook_ims/"

to_path = "notebook_ims/"

os.makedirs(to_path)

copy_tree(from_path,to_path)
import torch

import numpy as np

from torchvision import datasets

import torchvision.transforms as transforms



# convert data to torch.FloatTensor

transform = transforms.ToTensor()



# load the training and test datasets

train_data = datasets.MNIST(root='data', train=True,

                                   download=True, transform=transform)

test_data = datasets.MNIST(root='data', train=False,

                                  download=True, transform=transform)



# Create training and test dataloaders

num_workers = 0

# how many samples per batch to load

batch_size = 20



# prepare data loaders

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
import matplotlib.pyplot as plt

%matplotlib inline

    

# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy()



# get one image from the batch

img = np.squeeze(images[0])



fig = plt.figure(figsize = (5,5)) 

ax = fig.add_subplot(111)

ax.imshow(img, cmap='gray')
train_on_gpu = torch.cuda.is_available()
import torch.nn as nn

import torch.nn.functional as F



# define the NN architecture

class ConvDenoiser(nn.Module):

    def __init__(self):

        super(ConvDenoiser, self).__init__()

        ## encoder layers ##

        self.conv1 = nn.Conv2d(1,32,3,padding=1)

        self.conv2 = nn.Conv2d(32,4,3,padding=1)

        self.maxPool = nn.MaxPool2d(2,2)

        

        

        

        ## decoder layers ##

        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2

        self.conv3 = nn.Conv2d(4, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)

        

       





    def forward(self, x):

        ## encode ##

        x = self.maxPool(F.relu(self.conv1(x)))

        x = self.maxPool(F.relu(self.conv2(x)))

        ## decode ##

        x = F.upsample(x,scale_factor = 2,mode = "nearest")

        x = (F.relu(self.conv3(x)))

        x = F.upsample(x,scale_factor = 2,mode = "nearest")

        x = (F.relu(self.conv4(x)))

        

        return x



# initialize the NN

model = ConvDenoiser()

if train_on_gpu:

    model = model.cuda()

print(model)
# specify loss function

criterion = nn.MSELoss()



# specify loss function

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 20



# for adding noise to images

noise_factor=0.5



for epoch in range(1, n_epochs+1):

    # monitor training loss

    train_loss = 0.0

    

    ###################

    # train the model #

    ###################

    for data in train_loader:

        # _ stands in for labels, here

        # no need to flatten images

        images, _ = data

        

        

        

        ## add random noise to the input images

        noisy_imgs = images + noise_factor * torch.randn(*images.shape)

        # Clip the images to be between 0 and 1

        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

                

        if train_on_gpu:

            noisy_imgs = noisy_imgs.cuda()

            images = images.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        ## forward pass: compute predicted outputs by passing *noisy* images to the model

        outputs = model(noisy_imgs)

        # calculate the loss

        # the "target" is still the original, not-noisy images

        loss = criterion(outputs, images)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*images.size(0)

            

    # print avg training statistics 

    train_loss = train_loss/len(train_loader)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(

        epoch, 

        train_loss

        ))
# obtain one batch of test images

dataiter = iter(test_loader)

images, labels = dataiter.next()



# add noise to the test images

noisy_imgs = images + noise_factor * torch.randn(*images.shape)

noisy_imgs = np.clip(noisy_imgs, 0., 1.)



if train_on_gpu:

    noisy_imgs = noisy_imgs.cuda()

# get sample outputs

output = model(noisy_imgs)

# prep images for display

noisy_imgs = noisy_imgs.cpu().numpy()



# output is resized into a batch of iages

output = output.cpu().view(batch_size, 1, 28, 28)

# use detach when it's an output that requires_grad

output = output.detach().numpy()



# plot the first ten input images and then reconstructed images

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))



# input images on top row, reconstructions on bottom

for noisy_imgs, row in zip([noisy_imgs, output], axes):

    for img, ax in zip(noisy_imgs, row):

        ax.imshow(np.squeeze(img), cmap='gray')

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)