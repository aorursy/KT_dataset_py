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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F



%matplotlib inline


train_dir = '/kaggle/input/digit-recognizer/train.csv'

test_dir = '/kaggle/input/digit-recognizer/test.csv'
samp_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train_df = pd.read_csv(train_dir)

test_df = pd.read_csv(test_dir)


X_train, X_test, y_train, y_test = train_test_split(train_df.values[:,1:]/255,train_df.values[:,0] )
plt.figure()

plt.imshow(X_train[0].reshape(28,28))

plt.show()

y_train[0]
train_torch_x = torch.tensor(X_train, dtype=torch.float, requires_grad=True) #Used to train and test model

train_torch_y = torch.tensor(y_train, dtype=torch.long ) #Used to train and test model

test_torch_x = torch.tensor(X_test, dtype=torch.float) #Used to train and test model

test_torch_y = torch.tensor(y_test, dtype=torch.long) #Used to train and test model



sub_torch_x = torch.tensor(train_df.values[:,1:]/255, dtype=torch.float, requires_grad=True) #Used to train all data for submissions

sub_torch_y = torch.tensor(train_df.values[:,0], dtype=torch.long ) #Used to train all data for submissions



submission_x = torch.tensor(test_df.values/255, dtype=torch.float)
plt.imshow(train_torch_x[1].view(28,28).detach())

train_torch_y[1]
train_dataset = torch.utils.data.TensorDataset(train_torch_x,train_torch_y)
test_dataset = torch.utils.data.TensorDataset(test_torch_x,test_torch_y)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64)



testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.l1 = nn.Linear(784, 256)

        self.l2 = nn.Linear(256, 128)

        self.l3 = nn.Linear(128, 64)

        self.l4 = nn.Linear(64, 10)

        

    def forward(self, x):

        

        x = F.relu(self.l1(x))

        x = F.relu(self.l2(x))

        x = F.relu(self.l3(x))

        x = self.l4(x)

        

        return x
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
#This can be used, but it takes longer than others

epochs = 2



for e in range(epochs):

    for images, labels in trainloader:



        optimizer.zero_grad()



        output = model.forward(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        

    else:

        print(f"Loss: {loss.item()}")
#Used to train the model for testing here

epochs = 5

batch_size = 64

for e in range(epochs):

    for i in range(0,len(train_torch_y), batch_size):

        batch_x = train_torch_x[i:i+batch_size] 

        batch_y = train_torch_y[i:i+batch_size]

        optimizer.zero_grad()



        output = model.forward(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        

    else:

        print(f"Loss: {loss}")
#Testing the model



with torch.no_grad():

    correct = 0

    total = test_torch_y.shape[0]

    for i in range(0,len(test_torch_y), batch_size):

        

        test_batch_x = test_torch_x[i:i+batch_size]



        preds = model.forward(test_batch_x)

        

        pred = preds.argmax(dim=1)

        

        correct += (pred == test_torch_y[i:i+batch_size]).sum().item()

        

    print(f'Accuracy: {correct/total*100}')
#For training for submissions, using all the data

epochs = 10

batch_size = 64



for e in range(epochs):

    for i in range(0,len(sub_torch_y), batch_size):

        batch_x = sub_torch_x[i:i+batch_size] 

        batch_y = sub_torch_y[i:i+batch_size]

        optimizer.zero_grad()



        output = model.forward(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        

    else:

        print(f"Loss: {loss}")
with torch.no_grad():

    samp_sub['Label'] = model(submission_x).argmax(dim=1)
samp_sub.to_csv('submission.csv', index=False)