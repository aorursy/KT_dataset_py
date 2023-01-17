import os

print(os.listdir("../input"))
from os import walk

for (dirpath, dirnames, filenames) in walk("../input/"):

    print("Directory path: ", dirpath)

    print("Folder name: ", dirnames)

#     print("File name: ", filenames)
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torchvision import datasets, transforms



# import helper  # helper function is defined below, no need for importing
# define helper.py 

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True





def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax





def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')



def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    elif version == "Fashion":

        ax2.set_yticklabels(['T-shirt/top',

                            'Trouser',

                            'Pullover',

                            'Dress',

                            'Coat',

                            'Sandal',

                            'Shirt',

                            'Sneaker',

                            'Bag',

                            'Ankle Boot'], size='small');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
# Define default PATH

PATH = '../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data'
# data_dir = 'Cat_Dog_data/train'

data_dir = PATH + '/train' # load from Kaggle



transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()

                               ])# TODO: compose transforms here

dataset = datasets.ImageFolder(data_dir, transform=transform) # TODO: create the ImageFolder

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader
data_dir
# Run this to test your data loader

images, labels = next(iter(dataloader))

# helper.imshow(images[0], normalize=False)

imshow(images[0], normalize=False)
PATH
# data_dir = 'Cat_Dog_data'

data_dir = PATH



# TODO: Define transforms for the training data and testing data

train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                      transforms.RandomResizedCrop(224),

                                      transforms.RandomHorizontalFlip(),

                                      transforms.ToTensor()])



test_transforms = transforms.Compose([transforms.RandomRotation(30),

                                     transforms.RandomResizedCrop(224),

                                     transforms.ToTensor()])





# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
# change this to the trainloader or testloader 

data_iter = iter(testloader)



images, labels = next(data_iter)

fig, axes = plt.subplots(figsize=(10,4), ncols=4)

for ii in range(4):

    ax = axes[ii]

#     helper.imshow(images[ii], ax=ax, normalize=False)

    imshow(images[ii], ax=ax, normalize=False)
# Optional TODO: Attempt to build a network to classify cats vs dogs from this dataset