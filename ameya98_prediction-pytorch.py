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
import torch

import pandas as pd

import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import seaborn as sns

from torch import optim

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/Admission_Predict.csv')
df.head()
df.isnull().sum()
df.drop('Serial No.', inplace=True, axis=1)

df.rename({'GRE Score':'Gre_Score','TOEFL Score':'TOEFL_Score','Chance of Admit ': 'Chance_of_Admit', 'LOR ':'LOR','University Rating':'University_Rating'}, axis=1, inplace=True)
df.describe()
sns.heatmap(df.corr(), annot=True).set_title('Correlation Factors Heat Map', color='black', size='30')
cols=['Gre_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Research']

x=df[cols]

y=df[['Chance_of_Admit']]
from sklearn.model_selection import train_test_split

x_1, x_test, y_1, y_test = train_test_split(x, y, test_size=0.1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_1_mod=scaler.fit_transform(x_1)

x_test_mod=scaler.transform(x_test)
x_train_mod, x_val_mod, y_train, y_val = train_test_split(x_1_mod, y_1, test_size=0.1)
import torch.utils.data as data_utils

import torchvision.transforms as transforms





train = data_utils.TensorDataset(torch.Tensor(x_train_mod) , torch.Tensor(y_train.values.reshape(-1)))

train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
test = data_utils.TensorDataset(torch.Tensor(x_test_mod) , torch.Tensor(np.array(y_test.values.reshape(-1))))

test_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
val = data_utils.TensorDataset(torch.Tensor(x_val_mod) , torch.Tensor(y_val.values.reshape(-1)))

val_loader = data_utils.DataLoader(val, batch_size=50, shuffle=True)
class Network(nn.Module):

  def __init__(self):

    super().__init__()

    self.hidden1 = nn.Linear(7, 8)

    self.hidden2 = nn.Linear(8, 5)

    self.output = nn.Linear(5, 1)

    self.dropout = nn.Dropout(0.2)

        

  def forward(self, x):

    x=self.hidden1(x)

    x=self.dropout(x)

    x=self.hidden2(x)

    x=self.dropout(x)

    x=self.output(x)

    return x
model=Network()

model
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)
n_epochs = 200

train_losses=[]

val_losses=[]



# initialize tracker for minimum validation loss

valid_loss_min = np.Inf # set initial "min" to infinity



for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training

    for data, target in train_loader:

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval() # prep model for evaluation

    for data, target in val_loader:

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update running validation loss 

        valid_loss += loss.item()*data.size(0)

        

    # print training/validation statistics 

    # calculate average loss over an epoch

    train_loss = train_loss/len(train_loader.dataset)

    valid_loss = valid_loss/len(val_loader.dataset)

    train_losses.append(train_loss)

    val_losses.append(valid_loss)

    

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch+1, 

        train_loss,

        valid_loss

        ))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss
model.load_state_dict(torch.load('model.pt'))
plt.plot(train_losses,label='Training Label')

plt.plot(val_losses,label='Validation Label')

plt.legend(frameon=False)


# initialize lists to monitor test loss and accuracy

test_loss = 0.0







model.eval() # prep model for evaluation



for data, target in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)







# calculate and print avg test loss

test_loss = test_loss/len(test_loader.dataset)



print('Test Loss: {:.6f}\n'.format(test_loss))


