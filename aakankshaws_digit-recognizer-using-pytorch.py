# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import libraries

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

from torch import optim

from torchvision import transforms
# Prepare Dataset

# load data

train = pd.read_csv(r"../input/train.csv",dtype = np.float32)



# split data into features(pixels) and labels(numbers from 0 to 9)

targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization



# train test split. Size of train data is 80% and size of test data is 20%. 

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



# create feature and targets tensor for test set.

featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long



# batch_size, epoch and iteration

batch_size = 100

n_iters = 10000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)





dataiter = iter(train_loader)

images, labels = dataiter.next()

print(type(images))

print(images.shape)

print(labels.shape)
# visualize one of the images in data set

plt.imshow(features_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[10]))

plt.savefig('graph.png')

plt.show()

#import libraries and create model architecture

from torch import nn, optim

import torch.nn.functional as F



class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)

        

    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)

        

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x
model = Classifier()



images, labels = next(iter(test_loader))

# Get the class probabilities

ps = torch.exp(model(images))

# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples

print(ps.shape)
top_p, top_class = ps.topk(1, dim=1)

# Look at the most likely classes for the first 10 examples

print(top_class[:10,:])
#equate topclass and labels

equals = top_class == labels.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))

print(f'Accuracy: {accuracy.item()*100}%')
model = Classifier()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 30

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            for images, labels in test_loader:

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)



        # Dropout module with 0.2 drop probability

        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))



        # output so no dropout here

        x = F.log_softmax(self.fc4(x), dim=1)



        return x
model = Classifier()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 30

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            model.eval()

            for images, labels in test_loader:

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
def plot(img,ps):

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ps = ps.data.numpy().squeeze()

    

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    ax2.set_yticklabels(np.arange(10))

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)

    

    plt.tight_layout()
images, labels = next(iter(train_loader))



img = images[0].view(1, 784)

# Turn off gradients to speed up this part

with torch.no_grad():

    logps = model.forward(img)



# Output of the network are logits, need to take softmax for probabilities

ps = torch.exp(logps)

plot(img,ps)
#predictions of first 10 images

ps = torch.exp(model(images))

top_p, top_class = ps.topk(1,dim=1)

print(top_class[:10,:])
#equate topclass and labels

equals = top_class == labels.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))

print(f'Accuracy: {accuracy.item()*100}%')