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



trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor).view(-1,1,28,28)

trn_y_torch = torch.from_numpy(trn_y)



val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor).view(-1,1,28,28)

val_y_torch = torch.from_numpy(val_y)



test_torch = torch.from_numpy(TEST).type(torch.FloatTensor).view(-1,1,28,28)



#Create a dataset(combination of x and y) from torch tensors.



trn = TensorDataset(trn_x_torch,trn_y_torch)

val = TensorDataset(val_x_torch,val_y_torch)

test = TensorDataset(test_torch)



#create dataloader from datasets.



trn_dataloader = torch.utils.data.DataLoader(trn,batch_size=100,shuffle=False, num_workers=4)

val_dataloader = torch.utils.data.DataLoader(val,batch_size=100,shuffle=False, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test,batch_size=100,shuffle=False, num_workers=4)
# Formula to calculate shape as we go through layer by layer = [(X - F + 2P)/S] + 1

# Here,

# X = Width / Height

# F = Kernel size

# P = Padding

# S = Strides (default = 1)



#Our input to the first layer is going to be [batchsize,1,28,28]

#substitute, =[(28 - 5 + 2(0))/1] + 1

#             =[(23)/1] + 1

#             =23 + 1

#             =24

#                :: shape = [batchsize,output_nodes,24,24]

#               :: in this case [64,16,24,24] where 64 is the batch size and [16,24,24] is the shape of the tensor.

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1,16,5) #(channels,output,kernel_size)   [Batch_size,1,28,28]  --> [Batch_size,16,24,24]

        self.mxp1 = nn.MaxPool2d(2)   #                                 [Batch_size,16,24,24] --> [Batch_size,16,24/2,24/2] --> [Batch_size,16,12,12]

        self.conv2 = nn.Conv2d(16,24,5) #                               [Batch_size,16,12,12] --> [Batch_size,24,8,8]

        self.mxp2 = nn.MaxPool2d(2)   #                                 [Batch_size,24,8,8] ---> [Batch_size,32,8/2,8/2] ---> [Batch_size,24,4,4]

        self.linear1 = nn.Linear(24 * 4 * 4, 100)                       #input shape --> 100 outputs

        self.linear2 = nn.Linear(100,10)                                #100 inputs --> 10 outputs

        

    def forward(self,x):

        X = self.mxp1(F.relu(self.conv1(x)))

        X = self.mxp2(F.relu(self.conv2(X)))

        X = X.view(-1, 24 * 4 * 4)  #reshaping to input shape

        X = F.relu(self.linear1(X))

        X = self.linear2(X)

        return F.log_softmax(X, dim=1)

 

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