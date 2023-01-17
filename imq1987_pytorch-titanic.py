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
from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

y_train = data_train['Survived']

x_train = data_train.loc[:,data_train.columns != 'Survived']
object_features = []

for col in x_train.columns:

    if x_train[col].dtype == 'object':

        x_train[col + '_cat'] = x_train[col].astype('category').cat.codes

        object_features.append(col)
x_train.drop(object_features, axis = 1, inplace = True)



imp = Imputer()

x_train = imp.fit_transform(x_train)
data_test
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')

x_test = data_test



object_features = []

for col in x_test.columns:

    if x_test[col].dtype == 'object':

        x_test[col + '_cat'] = x_test[col].astype('category').cat.codes

        object_features.append(col)



x_test.drop(object_features, axis = 1, inplace = True)



imp = Imputer()

x_test = imp.fit_transform(x_test)
import torch

from torch.utils.data import Dataset, DataLoader

import torch.optim as torch_optim

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models
class TitancDataset(Dataset):

    def __init__(self, data, y):

        self.x_train = data

        self.y_train = y

        

        self.x_train = self.x_train

        self.y_train = np.array(self.y_train)

        

    def __len__(self):

        return len(self.x_train)

    

    def __getitem__(self, idx):

        return (self.x_train[idx], self.y_train[idx])
class TitancDataset_test(Dataset):

    def __init__(self, data):

        self.x_test = data

        

        self.x_test = self.x_test

        

    def __len__(self):

        return len(self.x_test)

    

    def __getitem__(self, idx):

        return (self.x_test[idx], _)
dataset_train = TitancDataset(x_train, y_train)
iterdata = iter(dataset_train)
next(iterdata)
trainloader = DataLoader(dataset_train, batch_size = 64, num_workers = 0, shuffle = True)
dataset_test = TitancDataset_test(x_test)

testloader = DataLoader(dataset_test, batch_size = 64, num_workers = 0, shuffle = False)
# Simple Neural network 

input_size = 11

hidden_size = 128

num_classes = 1 

num_epochs = 5

learning_rate = 0.001

BATCH_SIZE_1 = 101 #train_loader as it has 404 observations

BATCH_SIZE_2 = 51 #test_loader as it has 102 observations
device = "cuda" if torch.cuda.is_available() else "cpu"

kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
class LinearModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(LinearModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()

                           

    def get_weights(self):

        return self.weight

    

    def forward(self,x):

        out = self.fc1(x)

        out = self.relu(out)

        out = F.sigmoid(self.fc2(out)) #sigmoid as we use BCELoss

        return out
def train(model, device, train_loader, optimizer):

    model.train()

    y_true = []

    y_pred = []

    for i in train_loader:

        

        #LOADING THE DATA IN A BATCH

        data, target = i

 

        #MOVING THE TENSORS TO THE CONFIGURED DEVICE

        data, target = data.to(device), target.to(device)

       

        #FORWARD PASS

        output = model(data.float())

#         loss = criterion(output, target.unsqueeze(1))

        loss = criterion(output, target.float())

        

        #BACKWARD AND OPTIMIZE

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        # PREDICTIONS 

        pred = np.round(output.detach())

        target = np.round(target.detach())             

        y_pred.extend(pred.tolist())

        y_true.extend(target.tolist())

        

    print("Accuracy on training set is" ,accuracy_score(y_true,y_pred))
#TESTING THE MODEL

def test(model, device, test_loader):

    #model in eval mode skips Dropout etc

    model.eval()

#     y_true = []

    y_pred = []

    

    # set the requires_grad flag to false as we are in the test mode

    with torch.no_grad():

        for i in test_loader:

            

            #LOAD THE DATA IN A BATCH

            data,target = i

            

            # moving the tensors to the configured device

            data, t = data.to(device), _

            

            # the model on the data

            output = model(data.float())

                       

            #PREDICTIONS

            pred = np.round(output)

#             target = target.float()

#             y_true.extend(target.tolist()) 

            y_pred.extend(pred.reshape(-1).tolist())

    

    return y_pred

            

#     print("Accuracy on test set is" , accuracy_score(y_true,y_pred))

#     print("***********************************************************")
# Creating model and setting loss and optimizer.

model = LinearModel(input_size, hidden_size, num_classes).to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100

for epoch in range(num_epochs):

        train(model,device,trainloader,optimizer)

#         test(model,device,test_loader)
y_pred = test(model,device,testloader)
from collections import Counter



Counter(y_pred)
my_submission = pd.DataFrame({'PassengerId' :data_test['PassengerId'], 'Survived': y_pred})

my_submission.to_csv('submission.csv', index=False)