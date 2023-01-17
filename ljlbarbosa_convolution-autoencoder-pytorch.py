import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler



from torch.utils.data import DataLoader

from torchvision import datasets, transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# convert data to torch.FloatTensor

transform = transforms.ToTensor()



# load the training and test datasets

train_data = datasets.CIFAR10(root='data', train=True,

                                   download=True, transform=transform)

test_data = datasets.CIFAR10(root='data', train=False,

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



# helper function to un-normalize and display an image

def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    

# specify the image classes

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',

           'dog', 'frog', 'horse', 'ship', 'truck']
# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(classes[labels[idx]])
rgb_img = np.squeeze(images[3])

channels = ['red channel', 'green channel', 'blue channel']



fig = plt.figure(figsize = (36, 36)) 

for idx in np.arange(rgb_img.shape[0]):

    ax = fig.add_subplot(1, 3, idx + 1)

    img = rgb_img[idx]

    ax.imshow(img, cmap='gray')

    ax.set_title(channels[idx])

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            val = round(img[x][y], 2) if img[x][y] !=0 else 0

            ax.annotate(str(val), xy=(y,x),

                    horizontalalignment='center',

                    verticalalignment='center', size=8,

                    color='white' if img[x][y]<thresh else 'black')
import torch.nn as nn

import torch.nn.functional as F



# define the NN architecture

class ConvAutoencoder(nn.Module):

    def __init__(self):

        super(ConvAutoencoder, self).__init__()

        ## encoder layers ##

        # conv layer (depth from 3 --> 16), 3x3 kernels

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  

        # conv layer (depth from 16 --> 4), 3x3 kernels

        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2

        self.pool = nn.MaxPool2d(2, 2)

        

        ## decoder layers ##

        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)



    def forward(self, x):

        ## encode ##

        # add hidden layers with relu activation function

        # and maxpooling after

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        # add second hidden layer

        x = F.relu(self.conv2(x))

        x = self.pool(x)  # compressed representation

        

        ## decode ##

        # add transpose conv layers, with relu activation function

        x = F.relu(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)

        x = F.sigmoid(self.t_conv2(x))

                

        return x



# initialize the NN

model = ConvAutoencoder()

print(model)
# specify loss function

criterion = nn.BCELoss()



# specify loss function

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 100



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

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        outputs = model(images)

        # calculate the loss

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



# get sample outputs

output = model(images)

# prep images for display

images = images.numpy()





# output is resized into a batch of iages

output = output.view(batch_size, 3, 32, 32)

# use detach when it's an output that requires_grad

output = output.detach().numpy()



# # plot the first ten input images and then reconstructed images

# fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))



# # input images on top row, reconstructions on bottom

# for images, row in zip([images, output], axes):

#     for img, ax in zip(images, row):

#         ax.imshow(np.squeeze(img))

#         ax.get_xaxis().set_visible(False)

#         ax.get_yaxis().set_visible(False)



# plot the first ten input images and then reconstructed images

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(output[idx])

    ax.set_title(classes[labels[idx]])

    

# plot the first ten input images and then reconstructed images

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(classes[labels[idx]])