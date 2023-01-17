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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader
path = '../input/fruits/fruits-360'

print('Folders :', os.listdir(path))

classes = os.listdir(path + "/Training")

print('131 classes :', classes)
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor

dataset = ImageFolder(path + '/Training', transform=ToTensor())

print('Size of training dataset :', len(dataset))

test = ImageFolder(path + '/Test', transform=ToTensor())

print('Size of test dataset :', len(test))
img, label = dataset[5000]

print(img.shape)
def show_image(img, label):

    print('Label: ', dataset.classes[label], "("+str(label)+")")

    plt.imshow(img.permute(1, 2, 0))
show_image(img, label)
# batch_size, epoch and iteration

batch_size = 100

n_iters = 5000

num_epochs = n_iters / (len(dataset) / batch_size)

num_epochs = int(num_epochs)
# data loader

train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)
from torchvision.utils import make_grid

for images, labels in train_loader:

    fig, ax = plt.subplots(figsize=(14,14))

    ax.set_xticks([]); ax.set_yticks([])

    ax.imshow(make_grid(images, nrow=10).permute(1, 2, 0))

    break
# Create ANN Model

class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(ANNModel, self).__init__()

        # Linear function 1: 30000 --> 150

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity 1

        self.relu1 = nn.ReLU()

        # Linear function 2: 150 --> 150

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Non-linearity 2 : Tanh

        self.tanh2 = nn.Tanh()

        # Linear function 3: 150 --> 150

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Non-linearity 3 ReLU

        self.relu3 = nn.ELU()

        # Linear function 4 (readout): 150 --> ?????????

        self.fc4 = nn.Linear(hidden_dim, output_dim)

        

    def forward(self, x):

        # Linear function 1

        out = self.fc1(x)

        # Non-linearity 1

        out = self.relu1(out)

        # Linear function 2

        out = self.fc2(out)

        # Non-linearity 2

        out = self.tanh2(out)

        # Linear function 2

        out = self.fc3(out)

        # Non-linearity 2

        out = self.relu3(out)

        # Linear function 4 (readout)

        out = self.fc4(out)

        return out

# instantiate ANN

input_dim = 3*100*100

#hidden layer dim is one of the hyper parameter and it should be chosen and tuned.

hidden_dim = 150

output_dim = 131

# Create ANN

model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss

error = nn.CrossEntropyLoss()

# SGD Optimizer

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# ANN model training

count = 0

loss_list = []

iteration_list = []

accuracy_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 3*100*100))

        labels = Variable(labels)

        # Clear gradients

        optimizer.zero_grad()

        # Forward propagation

        outputs = model(train)

        # Calculate softmax and cross entropy loss

        loss = error(outputs,labels)

        # Calculating gradients

        loss.backward()

        # Update parameters

        optimizer.step()

        count += 1

        

        if count % 50 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0



            # Predict test dataset

            for images, labels in test_loader:



                test = Variable(images.view(-1, 3*100*100))

                

                # Forward propagation

                outputs = model(test)

                

                # Get predictions from the maximum value

                predicted = torch.max(outputs.data, 1)[1]

                

                # Total number of labels

                total += len(labels)



                # Total correct predictions

                correct += (predicted == labels).sum()

            

            accuracy = 100 * correct / float(total)

            

            # store loss and iteration

            loss_list.append(loss.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)



        if count % 500 == 0:

            # Print Loss

            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
# visualization loss

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("ANN: Loss vs Number of iteration")

plt.show()
# visualization accuracy

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("ANN: Accuracy vs Number of iteration")

plt.show()