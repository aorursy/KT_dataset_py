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
# Import packages

import numpy as np 

import pandas as pd

import os

import torch

from torchvision import datasets, transforms

from torch import optim

import torch.nn.functional as F

import matplotlib.pyplot as plt

%matplotlib inline

import torchvision

import seaborn as sns

sns.set_style("darkgrid")

from torchvision import models

import torch.nn as nn

import shutil

from PIL import Image

from sklearn.metrics import f1_score, confusion_matrix

import warnings

warnings.filterwarnings("ignore")
# Check for GPU compatibility

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

print(device)
train_dir = "../input/cleaned-data/data/data/train"

valid_dir = "../input/cleaned-data/data/data/validation"

test_dir = "../input/cleaned-data/data/data/test"
# training data transformations

transform_train = transforms.Compose([transforms.RandomResizedCrop(299),

                                     transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                     std=[0.229, 0.224, 0.225])])



# test data transformations

transform_test = transforms.Compose([transforms.RandomResizedCrop(299),

                                     transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])])
# batch size

batch_size = 16



# load train data

trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)



# load validation data

valset = datasets.ImageFolder(root=valid_dir, transform=transform_train)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
classes = list(trainset.class_to_idx.keys())

dataiter = iter(trainloader)

images, labels = dataiter.next()



print(images.shape)

print(labels.shape)
## Visualize single image



img = images[2]



# convert torch tensor to numpy array

npimg = img.numpy()



# transpose to get it in (n_H, n_W, n_C) format

npimg = np.transpose(npimg, (1,2,0))



# plot

plt.imshow(npimg)

plt.show()
## Visualize grid of images



def imshow(img):

    # convert to NumPy

    npimg = img.numpy() 

    plt.imshow(np.transpose(npimg, (1,2,0)))

    plt.show()

    

imshow(torchvision.utils.make_grid(images))

print(' '.join(classes[labels[j]] for j in range(batch_size)))
inception = models.inception_v3(pretrained=True)

print(inception)
# Freeze "features" network parameters

for param in inception.parameters():

    param.requires_grad = False

    

## Last layer has output containing 5 classes

inception.fc = nn.Linear(2048, 5)
def evaluate(dataloader, model):

    tp, fp, fn = 0, 0, 0

    # total, correct = 0, 0

    # iterate over every mini-batch

    for data in dataloader:

        # divide into features and labels

        inputs, labels = data

        # move them to device

        inputs, labels = inputs.to(device), labels.to(device)

        # make predictions

        outputs = model.forward(inputs)

        _, pred = torch.max(outputs[0].data, 1)

        # num of data points

        # total += labels.size(0)

        # switch to CPU

        pred = pred.cpu().detach().numpy()

        labels = labels.cpu().detach().numpy()

        # confusion matrix

        cm = confusion_matrix(labels, pred)

        # false positives

        FP = cm.sum(axis=0) - np.diag(cm)

        # false negatives

        FN = cm.sum(axis=1) - np.diag(cm)

        # true positives

        TP = np.diag(cm)

        # true negatives

        TN = cm.sum() - (FP + FN + TP)

        tp += sum(TP)

        fp += sum(FP)

        fn += sum(FN)



    # f1 score

    f1 = (2*tp) / ((2*tp + fp + fn))

    return f1
# migrate model to GPU

inception = inception.to(device)
## Loss function and Optimizer

loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.SGD(inception.parameters(), lr=0.0001, weight_decay=0.00001)
# loss_arr, loss_epoch_arr, max_epochs = [], [], 50



# for epoch in range(max_epochs):

#     # load training data in batches

#     i = 0

#     for data in trainloader:

#         # read data and labels

#         inputs, labels = data

#         inputs, labels = inputs.to(device), labels.to(device)

#         # make gradient zero

#         opt.zero_grad()

#         # forward prop

#         outputs = inception.forward(inputs)

#         # calculate loss

#         loss = loss_fn(outputs[0], labels)

#         # backward prop

#         loss.backward()

#         # weight update

#         opt.step()

#         # update loss for the batch

#         loss_arr.append(loss.item())

#         # delete cache

#         del inputs, labels, outputs

#         # update i

#         i += 1

#         if not i%100:

#             print("Loss at {} number mini-batch of epoch {} is {}".format(i, epoch, loss.item()))

#             print("#"*100)

#     # update loss over epoch

#     loss_epoch_arr.append(loss.item())

#     # train and test validation F-1 scores

#     print("Training F-1 score is", evaluate(trainloader, inception))

#     print('='*100)

#     print("Validation F-1 score is", evaluate(valloader, inception))

#     print('*'*100)