import scipy.special as sps

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def exactSolution(beta,n,j):

    zExact = 0.0;

    for m in range(0,n):

        zExact += 2*sps.binom(n-1,m)*np.exp(-beta*j*(n-1-2*m));



    return -np.log(zExact)/(beta*n);
print("At beta = 0.3, N = 20, J = 1, the free energy pr spin is F =")

print(exactSolution(0.3,20,1))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import json

from random import random as rnd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parameter import Parameter





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MaskedLinear(nn.Linear):

#This is a very inefficient implementation.

#We have a fully connected layer which we mask so that some weights

#are effectively zero.

    def __init__(self, n, in_channels, out_channels, exclusive, bias=True):

        super(MaskedLinear,self).__init__(n*in_channels, n*out_channels, bias)

        self.in_channels = in_channels

        self.out_channels = out_channels

        

        self.exclusive = exclusive

        self.mask = torch.ones(n,n)

        if self.exclusive:

            self.mask = torch.tril(self.mask,-1)

        else:

            self.mask = torch.tril(self.mask)

        self.mask = torch.cat([self.mask] * self.in_channels, dim = 1)

        self.mask = torch.cat([self.mask] * self.out_channels, dim = 0)

        

        self.weight.data *= self.mask

        self.weight.data *= torch.sqrt(self.mask.numel()/self.mask.sum())

        

        if torch.cuda.is_available():

            self.mask = self.mask.to(device)

        

        self.reset_parameters()

        

    def forward(self, x):

        return F.linear(x, self.mask * self.weight, self.bias)



    def extra_repr(self):

        return (super(MaskedLinear, self).extra_repr() +

                ', exclusive={exclusive}'.format(**self.__dict__))
class Net(nn.Module):

    def __init__(self,nspins):

        super(Net, self).__init__()

        self.nspins = nspins

        self.fc1 = MaskedLinear(self.nspins,1,20,True,False)

        self.fc2 = MaskedLinear(self.nspins,20,20,False,False)

        self.fc3 = MaskedLinear(self.nspins,20,20,False,False)

        self.fc4 = MaskedLinear(self.nspins,20,20,False,False)

        self.fc5 = MaskedLinear(self.nspins,20,20,False,False)

        self.fc6 = MaskedLinear(self.nspins,20,1,False,False)

        #Here we have two hidden layer and the output layer.

        

    def forward(self,sample):

        batchsize = sample.size()[0]

        with torch.no_grad():

            for i in range(0,self.nspins):

                x = torch.sigmoid(self.fc1(sample))

                x = torch.sigmoid(self.fc2(x))

                x = torch.sigmoid(self.fc3(x))

                x = torch.sigmoid(self.fc4(x))

                x = torch.sigmoid(self.fc5(x))

                x = torch.sigmoid(self.fc6(x))

                for j in range(0,batchsize):

                    if x[j,i] >= rnd():

                        sample[j,i] = 1

                    else:

                        sample[j,i] = -1

        #This is also very inefficient. In lieu of an actual ARNN implementation

        #in Pytorch we have fully connected layers. Here we actually evaluate

        #the net nspins more times than we have to!

        #If this wasn't a demonstration notebook I would change it, please don't

        #use this for anything important.

        x = torch.sigmoid(self.fc1(sample))

        x = torch.sigmoid(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))

        x = torch.sigmoid(self.fc4(x))

        x = torch.sigmoid(self.fc5(x))

        x = torch.sigmoid(self.fc6(x))

        

        return x
import torch.optim as optim



class Jorge():

#This class will train our network.

    def __init__(self,jj,beta,nspins,batchsize):

        self.jj = jj

        self.beta = beta

        self.nspins = nspins

        self.batchsize = batchsize

        self.net = Net(self.nspins)

        self.f = torch.zeros(self.batchsize,requires_grad=False)

        self.logprob = torch.zeros(self.batchsize)

        self.sample = torch.zeros(self.batchsize,self.nspins)

        if torch.cuda.is_available():

            self.net = self.net.to(device)

            self.f = self.f.to(device)

            self.logprob = self.logprob.to(device)

            self.sample = self.sample.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr = 1e-3, betas = (0.9, 0.99), eps = 1e-8)

    

    def objectives(self,x):

        qi = self.sample*(x-1/2) + 1/2

        q = torch.prod(qi,1)

        self.logprob = torch.log(q)

        

        with torch.no_grad():

            for j in range(0,self.batchsize):

                self.f[j] = 0

                for i in range(0,self.nspins-1):

                    self.f[j] = self.f[j] + self.sample[j,i]*self.sample[j,i+1]

                self.f[j] = self.f[j]*self.jj*self.beta + self.logprob[j]

    

    def step(self):

        self.optimizer.zero_grad()

        x = self.net(self.sample)

        self.objectives(x)

        self.loss = (self.f - self.f.mean()) * self.logprob

        self.loss_reinforce = self.loss.mean()

        self.loss_reinforce.backward()

        self.optimizer.step()

    

    def train(self,epochs,verbose=False):

        self.f_mean = torch.zeros(epochs,requires_grad=False)

        self.f_var = torch.zeros(epochs,requires_grad=False)

        self.loss_mean = torch.zeros(epochs,requires_grad=False)

        self.loss_var= torch.zeros(epochs,requires_grad=False)

        

        epoch = np.linspace(0,epochs-1,epochs,dtype=int)

        temps = np.linspace(0,self.beta,epochs)

        for i in epoch:

            self.step()

            with torch.no_grad():

                self.f_mean[i] = self.f.mean()/(self.nspins*self.beta)

                self.f_var[i] = self.f.var()/(self.nspins*self.beta)**2

                self.loss_mean[i] = self.loss.mean()

                self.loss_var[i] = self.loss.var()

                

            

        if verbose:

            plt.errorbar(epoch,self.f_mean.numpy(),np.sqrt(self.f_var.numpy()),None,'k',errorevery = 50)

            

            plt.xlabel('Epochs')

            plt.ylabel('F')

            

            plt.title('Machine Convergence')
beta = 0.3

n = 10

j = 1

batchsize = 1000



print("At beta = 0.3, N = 20, J = 1, the analytical free energy pr spin is F =")

print(exactSolution(beta,n,j))



trainer = Jorge(j,beta,n,batchsize)

epochs = 1000



trainer.train(epochs,verbose=True)
beta_min = 1

beta_max = 10

beta_step = 3

npoints = int(1+np.round((beta_max-beta_min)/beta_step))

beta_array = np.linspace(beta_min,beta_max,npoints)

f_mean_array = np.zeros(npoints)

f_std_array = np.zeros(npoints)

loss_mean_array = np.zeros(npoints)

loss_std_array = np.zeros(npoints)

f_exact = np.zeros(npoints)



n = 20

j = 1

batchsize = 1000

epochs = 2000



for i in range(0,npoints):

    beta = beta_array[i]

    trainer = Jorge(j,beta,n,batchsize)

    trainer.train(epochs,verbose=False)

    f_exact[i] = exactSolution(beta,n,j)

    f_mean_array[i] = trainer.f_mean[-1]

    f_std_array[i] = np.sqrt(trainer.f_var[-1])

    loss_mean_array[i] = trainer.loss_mean[-1]

    loss_std_array[i] = np.sqrt(trainer.loss_var[-1])

plt.figure(figsize=(9,9))



plt.subplot(311)

plt.errorbar(beta_array,f_mean_array,f_std_array,None,'k^')

plt.errorbar(beta_array,f_exact,None,None,'k')

plt.xlabel('beta')

plt.ylabel('F')

plt.legend(['ARNN','Exact'])

plt.subplot(312)

plt.errorbar(beta_array,loss_mean_array,loss_std_array,None,'k^')

plt.xlabel('beta')

plt.ylabel('loss')

plt.subplot(313)

plt.semilogy(beta_array,np.abs(f_mean_array-f_exact),'k^')

plt.xlabel('beta')

plt.ylabel('error')



plt.show()