#import required modules.



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.autograd import Variable



import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#normalization and preprocessing

X = train.iloc[:,1:].values / 255

X = (X-0.5)/0.5



Y = train.iloc[:,0].values



TEST = test.values / 255

TEST = (TEST-0.5)/0.5



print(X.shape,Y.shape,test.shape)
#split into train and validation.



trn_x,val_x,trn_y,val_y = train_test_split(X,Y,test_size=0.20)
#create torch tensor from numpy array



trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor)

trn_y_torch = torch.from_numpy(trn_y)



val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor)

val_y_torch = torch.from_numpy(val_y)



test_torch = torch.from_numpy(TEST).type(torch.FloatTensor)



#Create a dataset(combination of x and y) from torch tensors.



trn = TensorDataset(trn_x_torch,trn_y_torch)

val = TensorDataset(val_x_torch,val_y_torch)

test = TensorDataset(test_torch)



#create dataloader from datasets.



trn_dataloader = torch.utils.data.DataLoader(trn,batch_size=100,shuffle=False, num_workers=4)

val_dataloader = torch.utils.data.DataLoader(val,batch_size=100,shuffle=False, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test,batch_size=100,shuffle=False, num_workers=4)
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.linear1 = nn.Linear(784,250)  #input - 784 --> output --> 250

        self.linear2 = nn.Linear(250,100)  #input - 250(from previous layer output) --> output --> 100

        self.linear3 = nn.Linear(100,10)   #input - 100(from previous layer output) --> output --> 10(no of classes)

    

    def forward(self,X):

        X = F.relu(self.linear1(X))       #layer 1 - ReLu Activation

        X = F.relu(self.linear2(X))       #layer 2 - ReLu Activation

        X = self.linear3(X)               #layer 3

        return F.log_softmax(X, dim=1)    #return the output from layer 3 after applying log_softmax activation function.

 

m = Model()                             #init

print(m)                                #Model summary
#define our optimizer



optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
EPOCHS = 20

losses = []



m.train()



for epoch in range(EPOCHS):

    for batch_idx, (data, target) in enumerate(trn_dataloader):

        # Get Samples

        data, target = Variable(data), Variable(target)

                

        # Init

        optimizer.zero_grad()



        # Predict

        y_pred = m(data) 



        # Calculate loss

        loss = F.cross_entropy(y_pred, target)

        losses.append(loss.cpu().data.item())

        

        # Backpropagation

        loss.backward()

        optimizer.step()

        

        

        # Display

        if batch_idx % 100 == 1:

            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch+1,

                EPOCHS,

                batch_idx * len(data), 

                len(trn_dataloader.dataset),

                100. * batch_idx / len(trn_dataloader), 

                loss.cpu().data.item()), 

                end='')

import matplotlib.pyplot as plt



plt.plot(losses)
#validation



m.eval()



correct = 0



for batch_idx, (data, target) in enumerate(val_dataloader):



    data, target = Variable(data), Variable(target)



    output = m.forward(data)

        

    _, predicted = torch.max(output.data,1)

        

    for pred,target in zip(predicted,target):

        if pred == target:

            correct = correct+1

            

print("Acc:",correct * 100 / len(val_x_torch),"%","correct:",correct,"/",len(val_x_torch))
m.eval()



testset_predictions = []

for batch_id,image in enumerate(test_dataloader):

    image = torch.autograd.Variable(image[0])

    output = m(image)

    _, predicted = torch.max(output.data,1)

    for prediction in predicted:

        testset_predictions.append(prediction.item())

        

len(testset_predictions)
submission = pd.read_csv('../input/sample_submission.csv')

submission['Label'] = testset_predictions

submission.to_csv('mnist-pytorch.csv',index=False)