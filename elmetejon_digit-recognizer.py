# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



train_input = torch.tensor(train.values)

train_input.shape
test_input = torch.tensor(test.values)

test_input = torch.reshape(test_input, (-1, 28, 28))

test_input = test_input.unsqueeze(1)

test_input.shape
train_target = train_input[:,0]

train_input = train_input[:,1:]
plt.imshow(train_input[1].view(28,28), cmap="gray")

plt.show()

train_input = torch.reshape(train_input, (42000, 28, 28))

train_input = train_input.unsqueeze(1)
train_input.shape

test
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        

        #Define all the layers of the CNN

        self.conv1 = nn.Conv2d(1, 64, kernel_size = 5)

        self.bn1 = nn.BatchNorm2d(64)

        

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 5)

        self.bn2 = nn.BatchNorm2d(128)

        

        self.fc1 = nn.Linear(128, 100)

        self.out = nn.Linear(100, 10)

        self.dropout = nn.Dropout(p=0.5)

        

    def forward(self, x):

        

        x = F.max_pool2d(self.conv1(x), 3, 3)

        x = F.relu(x)

        x = self.bn1(x)

        

        x = F.max_pool2d(self.conv2(x), 4, 4)

        x = F.relu(x)

        x = self.bn2(x)

        x = self.dropout(F.relu(self.fc1(x.view(-1,128))))

        x = F.relu(self.out(x))

        

        return x

    

  


def train_model(model, train_input, train_target, mini_batch_size):

    criterion = nn.CrossEntropyLoss()

    

    eta = 1e-1

    optimizer = torch.optim.SGD(model.parameters(), lr = eta)

    sum_loss = 0

    

    for b in range(0, train_input.size(0), mini_batch_size):

        output = model(train_input.narrow(0, b, mini_batch_size))

        loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

        #loss = F.cross_entropy(input = train_target.narrow(0, b, mini_batch_size), target = output)

        model.zero_grad()

        loss.backward()

        optimizer.step()

        sum_loss += loss.item()

    return sum_loss







def compute_nb_errors(model, input, target, mini_batch_size):

    nb_errors = 0



    for b in range(0, input.size(0), mini_batch_size):

        output = model(input.narrow(0, b, mini_batch_size))

        _, predicted_classes = output.max(1)

        for k in range(mini_batch_size):

            if target[b + k] != predicted_classes[k]:

                nb_errors = nb_errors + 1

                

    return nb_errors
mini_batch_size= 100



model = CNN()

print(model)



for k in range(30):

    model.train

    train_model(model, train_input.float(), train_target, mini_batch_size)

    model.eval

    print(compute_nb_errors(model,train_input.float(), train_target, mini_batch_size)) #, compute_nb_errors(model,test_input,new_test_target, mini_batch_size))

    
output = model(test_input.float())
_,Predictions = output.max(1)

Predictions.detach().numpy()
#Predictions_pd = pd.Dataframe()



data = {'ImageId':np.arange(1,28001), 'label':Predictions.detach().numpy()}

output = pd.DataFrame(data)
output.to_csv('submission.csv', index =False)