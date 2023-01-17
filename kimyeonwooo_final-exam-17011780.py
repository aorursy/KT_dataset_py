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
import pandas as pd

import numpy as np

import torch

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import torch.nn.functional as F

import torch.nn as nn 

import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'



torch.manual_seed(777)

if device == 'cuda':

    torch.cuda.manual_seed_all(777)
train = pd.read_csv('../input/18011765watermelon-price/train_water_melon_price.csv')

print(train.head(10))

print(train.info())
test = pd.read_csv('../input/18011765watermelon-price/test_watermelon_price.csv')

print(test.head(10))

print(test.info())
learning_rate = 0.01

training_epoch = 2000

batch_size = 200
x_train = train.iloc[:,1:-1]

y_train = train.iloc[:,[-1]]



x_train = np.array(x_train)

y_train = np.array(y_train)



x_train = torch.FloatTensor(x_train)

y_train = torch.FloatTensor(y_train)



print(x_train.shape)

print(y_train.shape)
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)



data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size = batch_size,

                                          shuffle=True,

                                          drop_last=True)



class MishFunction(torch.autograd.Function):

    @staticmethod

    def forward(ctx, x):

        ctx.save_for_backward(x)

        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))



    @staticmethod

    def backward(ctx, grad_output):

        x = ctx.saved_variables[0]

        sigmoid = torch.sigmoid(x)

        tanh_sp = torch.tanh(F.softplus(x)) 

        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))



class Mish(nn.Module):

    def forward(self, x):

        return MishFunction.apply(x)



def to_Mish(model):

    for child_name, child in model.named_children():

        if isinstance(child, nn.ReLU):

            setattr(model, child_name, Mish())

        else:

            to_Mish(child)
linear1 = nn.Linear(7,14,bias=True) 

linear2 = nn.Linear(14,28,bias=True)

linear3 = nn.Linear(28,14,bias=True)

linear4 = nn.Linear(14,7,bias=True)

linear5 = nn.Linear(7,1,bias=True)

mish = Mish() # activation function

nn.init.kaiming_normal_(linear1.weight)

nn.init.kaiming_normal_(linear2.weight)

nn.init.kaiming_normal_(linear3.weight)

nn.init.kaiming_normal_(linear4.weight)

nn.init.kaiming_normal_(linear5.weight)

model = torch.nn.Sequential(

    linear1,mish,

    linear2,mish,

    linear3,mish,

    linear4,mish,

    linear5

).to(device)
loss = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
total_batch = len(data_loader)



for epoch in range(training_epoch):

    avg_cost = 0

    for X,Y in data_loader:

        X = X.to(device)

        Y = Y.to(device)



        optimizer.zero_grad()

        hypothesis = model(X)

        cost = loss(hypothesis,Y)

        cost.backward()

        optimizer.step()





        avg_cost += cost/total_batch

    

    if epoch % 10 == 0:  

        print('Epoch:', '%d' % (epoch ), 'Cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished')
test_data = test.iloc[:,1:]

test_data = np.array(test_data)

test_data = torch.FloatTensor(test_data).to(device)



with torch.no_grad():

    predict = model(test_data)



predict
correct_prediction = predict.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('../input/18011765watermelon-price/submit_sample.csv')

for i in range(len(correct_prediction)):

  submit['Expected'][i]=correct_prediction[i].item()

submit.to_csv('submit.csv', mode = 'w', index = False, header = True)

submit
