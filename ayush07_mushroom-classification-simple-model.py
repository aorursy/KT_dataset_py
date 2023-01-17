# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read data

data = pd.read_csv('../input/mushrooms.csv')

data.head()
#Convert to numpy array

dataArray = np.array(data)

total = len(dataArray)

N = 7000
#Split into test and training sets

X_train = dataArray[:N,1:]

y_train = dataArray[:N,0]

X_test = dataArray[N:,1:]

y_test = dataArray[N:,0]
#Create a dic to convert each data point from char to int

dicts = []

codes = []

codes.append('bcxfks')

codes.append('fgys')

codes.append('nbcgrpuewy')

codes.append('tf')

codes.append('alcyfmnps')

codes.append('adfn')

codes.append('cwd')

codes.append('bn')

codes.append('knbhgropuewy')

codes.append('et')

codes.append('bcuezr?')

codes.append('fyks')

codes.append('fyks')

codes.append('nbcgopewy')

codes.append('nbcgopewy')

codes.append('pu')

codes.append('nowy')

codes.append('not')

codes.append('ceflnpsz')

codes.append('knbhrouwy')

codes.append('acnsvy')

codes.append('glmpuwd')

for code in codes:

    temp = {}

    for i in range(0,len(code)):

        temp[code[i]] = i

    dicts.append(temp)
#Number of features

m  = X_train.shape[1]
#Training data points char->int using dict

for j in range(0,m):

    for x in X_train:

        x[j] = dicts[j][x[j]]
#Test data points char->int using dict

for j in range(0,m):

    for x in X_test:

        x[j] = dicts[j][x[j]]
#Class Labels; Posionous - 0, Edible - 1

for i in range(0,len(y_train)):

    if y_train[i] == 'p':

        y_train[i] = 0

    else:

        y_train[i] = 1

        

for i in range(0,len(y_test)):

    if y_test[i] == 'p':

        y_test[i] = 0

    else:

        y_test[i] = 1
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
#Simple Model consisting of Affine and Softmax Layer, Cross Entropy Loss function thereafter 

model = nn.Sequential(nn.Linear(22,2),

                     nn.Softmax())



dType = torch.FloatTensor

model.type(dType)

lossFunc = nn.CrossEntropyLoss().type(dType)

#Optimized using Adam

optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)
#Converting to tensor objects (Training data)

X_train_torch = torch.FloatTensor(X_train)

X_train_var = torch.autograd.Variable(X_train_torch)

y_train_trch = torch.LongTensor(y_train)

y_train_var = torch.autograd.Variable(y_train_trch)
#Train

epochs = 500

for epoch in range(0,epochs):

    

    optimizer.zero_grad()

    forward = model(X_train_var)

    loss = lossFunc(forward,y_train_var)

    loss.backward()

    optimizer.step()

    

    print("epoch ",epoch, " loss = ", loss.data[0])
#Training data predictions

out_train = model(X_train_var)

_, pred_train = torch.max(out_train.data,1)

#print(pred_train.size(0))

#print(pred_train)

correct = (pred_train == y_train_trch).sum()

print("Total correct  = ",correct,"Train Accuracy = ", correct/len(X_train) )
#Converting to tensor objects (Test Data)

X_test_torch = torch.FloatTensor(X_test)

y_test_torch = torch.LongTensor(y_test)

X_test_var = torch.autograd.Variable(X_test_torch)
out_test = model(X_test_var)

_, pred_test = torch.max(out_test.data,1)

correct = (pred_test == y_test_torch).sum()

print("Total Correct = ",correct,"Test Accuracy = ", correct/len(y_test))