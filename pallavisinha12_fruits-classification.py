# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#imported libraries

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import copy

from PIL import Image



import torch

import torch.nn as nn

import torchvision

import torchvision.datasets as datasets

import torchvision.models as models

import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
#resizing image as per reqyirements of resnet

from torchvision import transforms

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
DIR_TRAIN = "../input/fruits/fruits-360/Training"

DIR_TEST = "../input/fruits/fruits-360/Test"

classes_train = os.listdir(DIR_TRAIN)

classes_test = os.listdir(DIR_TEST)

print(len(classes_test))
classes = sorted(classes_train)
trainset = torchvision.datasets.ImageFolder(root='../input/fruits/fruits-360/Training', transform=transform)

testset = torchvision.datasets.ImageFolder(root='../input/fruits/fruits-360/Test', transform=transform)
batch_size = 16

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()



print(images.shape)

print(images[0].shape)

print(labels.shape)
model = models.resnet50(pretrained = True)
print(model)
for p in model.parameters():

    p.requires_grad = False
#changed last sequential layer of model

model.fc = nn.Sequential(nn.Linear(2048, 1024),

                           nn.ReLU(),

                           nn.Linear(1024, 512),

                           nn.ReLU(),

                           nn.Linear(512, 131),

                           nn.LogSoftmax(dim=1))
model = model.to(device)
import torch.optim as optim

loss_fn = nn.NLLLoss()

opt = optim.Adam(model.parameters(), lr = 0.001)
def evaluation(dataloader):

    total, correct = 0, 0

    model.eval()

    for data in dataloader:

        inputs, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100 * correct / total
loss_epoch_arr = []

max_epochs = 2

min_loss = 1000

n_iters = np.ceil(50000/batch_size)

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        inputs,labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        model.train()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        if min_loss > loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(model.state_dict())

            print('Min loss %0.2f' % min_loss)

        

            if i % 100 == 0:

                print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

            

            del inputs, labels, outputs

            torch.cuda.empty_cache()

    loss_epoch_arr.append(loss.item())

    model.eval()

        

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (

        epoch+1, max_epochs, 

        evaluation(testloader), evaluation(trainloader)))

    

    

plt.plot(loss_epoch_arr)

plt.show()

model.load_state_dict(best_model)

print(evaluation(trainloader), evaluation(testloader))
dataiter = iter(testloader)

images, labels = dataiter.next()

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    

imshow(torchvision.utils.make_grid(images[:1]))

print("Ground truth is")

print(' '.join(classes_train[labels[j]] for j in range(1)))

images = images.to(device)

outputs = model(images)

max_values, pred_class = torch.max(outputs.data, 1)

print("Predicted label is")

print(' '.join(classes_train[pred_class[j]] for j in range(1)))