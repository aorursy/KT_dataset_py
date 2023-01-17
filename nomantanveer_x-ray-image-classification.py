# importing Libraries

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import os

import time

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from PIL import Image

from torch.optim import lr_scheduler

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import transforms, utils, datasets, models
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'



classes = ['NORMAL', 'PNEUMONIA']

# TODO: Define transforms for the training data and testing data

train_transforms = transforms.Compose([transforms.Resize(256),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomRotation(10),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
# Visualize some sample data



# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(10):

    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title(classes[labels[idx]])
model_pre = models.resnet50(pretrained=True)

remove_data_parallel = False
model_pre
for param in model_pre.parameters():

    param.requires_grad = False
from collections import OrderedDict

fc = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(2048, 1024)),

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(1024, 2)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))

    

model_pre.fc = fc
model_pre
# if GPU is available, move the model to GPU

if train_on_gpu:

    model_pre.cuda()
import torch.optim as optim



# specify loss function (categorical cross-entropy)

criterion = nn.NLLLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.001

optimizer = optim.Adam(model_pre.fc.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 1



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    

    # model by default is set to train

    for batch_i, (data, target) in enumerate(trainloader):

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model_pre(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss 

        train_loss += loss.item()

        

        if batch_i % 10 == 9:    # print training loss every specified number of mini-batches

            print('Epoch %d, Batch %d loss: %.16f' %

                  (epoch, batch_i + 1, train_loss / 10))

            train_loss = 0.0
# track test loss 

test_loss = 0.0

class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))



model_pre.eval() # eval mode



# iterate over test data

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model_pre(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update  test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(47):  #batch size

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# calculate avg test loss

test_loss = test_loss/len(testloader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))