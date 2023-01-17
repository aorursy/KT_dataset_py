# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import torch

from torch import nn, optim

import torch.utils.data as data_utils

import matplotlib.pyplot as plt





test_data = pd.read_csv('../input/digit-recognizer/test.csv')

train_data = pd.read_csv('../input/digit-recognizer/train.csv')
print('test image: ', test_data.shape[0])

print('train image: ', train_data.shape[0])

print('feature: ', test_data.shape[1])
# convert the data into tensor and normalize the values.

# For cross entropy loss function, the label values must be integer. (long())

features_tensor = torch.FloatTensor((train_data.drop(['label'], axis=1).values))/255

targets_tensor = torch.FloatTensor(train_data['label'].values).long()





print("train images size: ", features_tensor.shape[0],

      "\nfeatures size: ", features_tensor.shape[1],

      "\nlabels size: ", targets_tensor.shape[0])



# set train_loader with custom data (type: tensor)

train = data_utils.TensorDataset(features_tensor, targets_tensor)

train_loader = data_utils.DataLoader(train, batch_size= 512, shuffle=True)
# build a network

model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1))



# set the loss

criterion = nn.NLLLoss()

# set the optimizer / Stochastic Gradient Descent

optimizer = optim.SGD(model.parameters(), lr=0.03)

# set epoch

epoch = 20
losses = []



for e in range(epoch):

    total_loss = 0

        

    for images, labels in train_loader:

        # flatten images 

        images = images.view(images.shape[0], -1)

        

        # initialize the gradients

        optimizer.zero_grad()

        # calculate the loss

        output = model(images)

        loss = criterion(output, labels)

        

        total_loss += loss.item()

        

        # backpropagation

        loss.backward()

        # update the weight

        optimizer.step()

                

        

    else:

        epoch_loss = total_loss / len(train_loader)

        # stack the loss

        losses.append(epoch_loss)

        print("training loss: ", epoch_loss)         
%matplotlib inline

plt.plot(losses)

plt.ylabel('Training Loss')

plt.xlabel('training length')

plt.show()
import pandas as pd

import torch

from torch import nn, optim

import torch.utils.data as data_utils

import matplotlib.pyplot as plt

import torch.nn.functional as F





data = pd.read_csv('../input/digit-recognizer/train.csv')
# convert the data into tensor and normalize the values.

# For cross entropy loss function, the label values must be integer. (long())

features = torch.FloatTensor((data.drop(['label'], axis=1).values))/255

labels = torch.FloatTensor(data['label'].values).long()
# distribute 80 % of the data into the training set / 20 % of the data into the validation set.

train_size = int(0.8 * len(data))

val_size = len(data) - train_size



aug_data = data_utils.TensorDataset(features, labels)

train_set, val_set = torch.utils.data.random_split(aug_data, [train_size, val_size])
# set data loader

train_loader = data_utils.DataLoader(train_set, batch_size= 256, shuffle=True)

test_loader = data_utils.DataLoader(val_set, batch_size = 256, shuffle=True)
# build a feed-forward neural network.

class classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128,64)

        self.fc4 = nn.Linear(64, 10)

        

        self.dropout = nn.Dropout(p = 0.2)

        

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x
# set the configuration 

model = classifier()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.03)



epochs = 10

steps = 0

train_losses, val_losses = [], []
for e in range(epochs):

    running_loss = 0

    

    for images, labels in train_loader:

        

        optimizer.zero_grad() #initialize gradients

        

        output = model(images)

        loss = criterion(output, labels)

        loss.backward() # backprogagation

        optimizer.step() # update the weights

        

        running_loss += loss.item()

    

    else:

        accuracy = 0 

        val_loss = 0

        

        # temporarily turn off calculating gradients 

        # to efficiently compute tensors in the validation set.

        with torch.no_grad(): 

            # turn off dropout

            # I want to use the same model trained

            model.eval()

            for images, labels in test_loader:

                

                output = model(images)

                val_loss += criterion(output, labels)

                

                prob = torch.exp(output)

                top_prob, top_labels = prob.topk(1, dim=1)

                comparison = top_labels == labels.view(*top_labels.shape)

                accuracy += torch.mean(comparison.type(torch.FloatTensor)) 

                # torch.mean takes tensors only in float type.

                

            # after evaluating the accuracy for validation set,

            # turn on dropout for the next epoch

            model.train()

            

            train_losses.append(running_loss/len(train_loader))

            val_losses.append(val_loss/len(test_loader))

            

            print("Epoch: {}/{}..".format(e+1, epochs),

                 "Training Loss: {:.3f}..".format(train_losses[-1]),

                 "Test Loss: {:.3f}..".format(val_losses[-1]),

                 "Test Accuracy: {:.3f}..".format(accuracy/len(test_loader)))



                
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label="training loss")

plt.plot(val_losses, label='validation loss')

plt.legend(frameon=False)