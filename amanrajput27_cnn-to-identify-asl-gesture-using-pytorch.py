import pandas as pd

import numpy as np

import torch

from tqdm import tqdm

from random import shuffle



train_path = "/kaggle/input/sign-language-mnist/sign_mnist_train.csv"

train_csv = pd.read_csv(train_path)               #csv file read and stored in DataFrame object

train_list = train_csv.values.tolist()            #contents of DF object converted to list with each row as a list

train_data = []

for i in tqdm(train_list):                        #tqdm used for progress bar

    x = torch.Tensor(i[1:])

    x = x/255                                     #rescaling values of image tensor b/w 0-1 through division by one

    y = torch.Tensor(i[0:1])

    z = [x,y]

    train_data.append(z)

    

test_path = "/kaggle/input/sign-language-mnist/sign_mnist_test.csv"

test_csv = pd.read_csv(test_path)

test_list = test_csv.values.tolist()

test_data = []

for i in tqdm(test_list):

    x = torch.Tensor(i[1:])

    x = x/255

    y = torch.Tensor(i[0:1])

    z = [x,y]

    test_data.append(z)



shuffle(train_data)

shuffle(test_data)



if torch.cuda.is_available():

    device = torch.device("cuda:0")

    print("gpu")

else:

    device = torch.device("cpu")

    print("cpu")
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):                                           # class Net inherits from predefined Module class in torch.nn

    def __init__(self):                                         # calling constructor of  parent class

        super().__init__()                                     

        

        

        self.conv1 = nn.Conv2d(1,32,3)              # 2d convolution layer : (input : 1 image , output : 32 channels , kernel size : 3*3)

        self.conv2 = nn.Conv2d(32,64,3)

        self.conv3 = nn.Conv2d(64,128,3)

        

        self.linear_in = None                      # used to calculate input of first linear layer by passing fake data through 2d layers

        x = torch.rand(28,28).view(-1,1,28,28)     # using convs function

        self.convs(x)

    

        self.fc1 = nn.Linear(self.linear_in,512)

        self.fc2 = nn.Linear(512,26)

        

    def convs(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)) , (2,2) )      # relu used for activation function 

        x = F.max_pool2d(F.relu(self.conv2(x)) , (2,2) )      # max_pool2d for max pooling results of each kernel with window size 2*2

        x = F.max_pool2d(F.relu(self.conv3(x)) , (2,2) )

        

        if self.linear_in == None:

            self.linear_in = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]  # input of first linear layer is multiplication of dimensions of ouput 

        return x                                                        # tensor of the 2d layers

    

    def forward(self,x):                                    # forward pass function uses the convs function to pass through 2d layers

        x = self.convs(x)

        x = x.view(-1,self.linear_in)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        x = F.log_softmax(x ,dim = -1)                     # log_softmax for finding output neuron with highest value

        return x



net  = Net()    

net.to(device)                                            # for moving model over to gpu
import torch.optim as optim



loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr = 0.001)



for epoch in tqdm(range(10)):

    for i in (range(0,27400,100)):

        batch = train_data[i:i+100]

        

        batch_x = torch.FloatTensor(100,784)

        batch_y = torch.LongTensor(100,1)

        

        for j in range(100):

            batch_x[j] = batch[j][0]                         

            batch_y[j] = batch[j][1]

        

        batch_x = batch_x.view(100,1,28,28)

        batch_y = batch_y.view(100)

        

        batch_x =  batch_x.to(device)                   # for moving each batch to gpu

        batch_y =  batch_y.to(device)

      

        net.zero_grad()                                 # to make the gradients zero before calculating loss 

        outputs  = net(batch_x)

        loss = F.nll_loss(outputs , batch_y)

        loss.backward()                                 # backpropagation 

        optimizer.step()                                # adjusting parameters of model

    print(f"Epoch : {epoch} , Loss : {loss}")

 

        

correct = 0

total = 0

with torch.no_grad():                           # not calculating gradients for testing data

    for data in (test_data):

        x = data[0]

        x = x.view(-1,1,28,28)

        y = data[1]

        

        x = x.to(device)

        y = y.to(device)

        

        output = net(x)

        output = torch.argmax(output)

        if output == y:

            correct += 1

        total += 1

print("correct : " , correct)

print("total : " , total)

print("accuracy : " , round(correct/total , 3))