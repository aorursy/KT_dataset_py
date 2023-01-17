# Preparing the data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 



data=pd.read_csv('/kaggle/input/sandp500/all_stocks_5yr.csv')

data=data.pivot(index='date',columns='Name',values='close')

data=data.values

data=np.nan_to_num(data)



# Normalization of data

data= (data - np.min(data,axis=0) )/ ( np.max(data,axis=0) - np.min(data,axis=0) )
# Checking multivariate stuff : Just an exercise

from torch.distributions.multivariate_normal import MultivariateNormal

import torch

m=MultivariateNormal(torch.from_numpy(np.array([-0.9,-0.3,0.3,0.9]).astype(np.float32)), torch.eye(4))

m.sample(torch.Size([10,10])).size()
import os

import numpy as np

import math

import torch

import torchvision

from torch import nn

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms



from torch.nn.parameter import Parameter

import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal



num_epochs = 1000

learning_rate = 0.0001

torch.manual_seed(42)

%matplotlib inline



class LinearClippedWeight(nn.Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=False):

        super(LinearClippedWeight, self).__init__()

        self.in_features = in_features

        self.out_features = out_features

        #self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        #multiVar=MultivariateNormal(torch.from_numpy(np.array([-0.9,-0.3,0.3,0.9]).astype(np.float32)), torch.eye(4))

        #self.weight = nn.Parameter(multiVar.sample([out_features, in_features]))

        #print("Size of self.weight is {0}".format(self.weight.size()))

        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        #w = self.weight.data

        #w = w.clamp(0.2,0.8)

        #self.weight.data = w

        if bias:

            self.bias = nn.Parameter(torch.Tensor(out_features))

        else:

            self.register_parameter('bias', None)

        #self.reset_parameters()



    def reset_parameters(self):

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)

            bound = 1 / math.sqrt(fan_in)

            torch.nn.init.uniform_(self.bias, -bound, bound)



    def forward(self, input):

        # Clipping the weights

        #w=self.weight.data

        #weightData=w.detach().numpy().reshape(-1)

        #plt.hist(weightData,bins=10)

        #w[((w > 0.3) & (w < 0.7)) | ((w < -0.3) & (w > -0.7))]=0

        #w[(w >= 0.7) & (w <= 1)]=1 - ((w[(w >= 0.7) & (w <= 1)])/3)

        #w[(w >=0 ) & (w <=0.3)]=w[(w >=0 ) & (w <=0.3)]/3

        #w[(w >= -0.3) & (w <= 0)]=w[(w >= -0.3) & (w <= 0)]/3

        #w[w <= -0.7]=-1 + w[w <= -0.7]/3

        #w[(w < 0.3)  | (w > -0.3)]=0

        #self.weight.data=w

        #self.weight1 = self.weight.clone()

        #self.weight1 = self.weight1.clamp(0.3,0.7)

        #self.weight=nn.Parameter(self.weight1)

        #self.weight.data==self.weight.clamp(0.3,0.7)

        #self.weight1[((self.weight1 > 0.3) & (self.weight1 < 0.7)) | ((self.weight1 < -0.3) & (self.weight1 > -0.7))]=0

        #self.weight1[(self.weight1 >= 0.7) & (self.weight1 <= 1)]=1 - ((self.weight1[(self.weight1 >= 0.7) & (self.weight1 <= 1)])/3)

        #self.weight1[(self.weight1 >=0 ) & (self.weight1 <=0.3)]=self.weight1[(self.weight1 >=0 ) & (self.weight1 <=0.3)]/3

        #self.weight1[(self.weight1 >= -0.3) & (self.weight1 <= 0)]=self.weight1[(self.weight1 >= -0.3) & (self.weight1 <= 0)]/3

        #self.weight1[self.weight1 <= -0.7]=-1 + self.weight1[self.weight1 <= -0.7]/3

        #self.weight1[(self.weight1 < 0.3)  | (self.weight1 > -0.3)]=0

        

        #print("Type of weight1 is {0}".format(type(self.weight1)))

        #self.weight=nn.Parameter(self.weight1)

        #print("The current self.weight is {0}".format(self.weight))

        return F.linear(input, self.weight, self.bias)



    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}'.format(

            self.in_features, self.out_features, self.bias is not None

        )



class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder, self).__init__()

        self.linear1=LinearClippedWeight(505,100)

        self.r1=nn.ReLU(True)

        self.linear2=LinearClippedWeight(100,10)

        self.r2=nn.ReLU(True)

        self.linear3=nn.Linear(10,100)

        self.r3=nn.ReLU(True)

        self.linear4=nn.Linear(100,505)

        self.r4=nn.ReLU(True)

        

    def forward(self, x):

        x = self.r1(self.linear1(x))

        x = self.r2(self.linear2(x))

        x = self.r3(self.linear3(x))

        x = self.r4(self.linear4(x))

        return x



#w = w.clamp(-1,1)    



# Configuration

model = autoencoder()



criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)





# Data Prep

dataInput=Variable(torch.from_numpy(data.astype(np.float32)))



# Epochs

for epoch in range(num_epochs):

    dataOutput = model(dataInput)

    l1_reg = None

    l1_reg = model.linear1.weight.norm(1) + model.linear2.weight.norm(1)

    #for W in model.parameters():

    #    if l1_reg is None:

    #        l1_reg = W.norm(2)

    #    else:

    #        l1_reg = l1_reg + W.norm(2)

    loss = criterion(dataOutput, dataInput) + l1_reg

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, num_epochs, loss))
detachedWeights=model._modules['linear1'].weight.detach().numpy()

print(detachedWeights)