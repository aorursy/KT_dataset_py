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
#import pytorch and torchvision

import torch , torchvision

from torchvision import datasets , transforms
#define tranformation and normalization

#Normalize does the following for each channel: 

#image = (image - mean) / std The parameters mean, 

#std are passed as 0.5, 0.5 in your case. 

#This will normalize the image in the range [-1,1]. 

#For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, 

#the maximum value of 1 will be converted…

transform = transforms.Compose([transforms.ToTensor(),

                              transforms.Normalize((0.5),(0.5))])
#download and load the train data

print("downloading train data")

trainset = datasets.FashionMNIST('/F_mnist/' , download = True,

                                train = True , transform=transform)

print("loading train data")

trainload = torch.utils.data.DataLoader(trainset , batch_size=128 , 

                                       shuffle=True)



#download and load the test data

print("downloading test data")

testset = datasets.FashionMNIST('/F_mnist' , download=True , train= False,

                               transform=transform)

print("loading test data")

testload = torch.utils.data.DataLoader(testset , batch_size = 128,

                                      shuffle=True)
#defining the model



from torch import nn, optim

import torch.nn.functional as F





class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,128)

        self.fc4 = nn.Linear(128,64)

        self.fc5 = nn.Linear(64,10)

        # Dropout module with 0.35 drop probability

        self.dropout = nn.Dropout(p=0.35)

    def forward(self,x):

        #flatten the input tensor

        x = x.view(x.shape[0],-1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        #using elu activation function

        #exponential linear unit(variant of relu), slightly bent towards the negative axis

        x = self.dropout(F.elu(self.fc3(x)))

        x = self.dropout(F.elu(self.fc4(x)))

        #output

        x = F.log_softmax(self.fc5(x), dim = 1)

        

        return x

        

#initializing the model

model  = Classifier() 

#using nll loss

loss = nn.NLLLoss()

#optimizer

optmizer = optim.Adam(model.parameters() , lr = 0.001)

epochs = 70

steps = 0



train_losses , test_losses = [] , []

for e in range(epochs):

    runnning_loss = 0

    print("running  train epoch ",e)

    #performing training(forward and backward pass)

    for images , labels in trainload:

        #restoring optimizer state

        optmizer.zero_grad()

        #getting output of forward pass

        sof_log = model(images)

        #calculating the loss

        loss_ = loss(sof_log,labels)

        #performing backward pass

        loss_.backward()

        #updating the parameters

        optmizer.step()

        

        #adding loss

        runnning_loss += loss_.item()

    else:  #performing test 

        test_loss = 0

        accuracy = 0

        

        #turn off gradients for testing, as no backpropagation is needed

        #turning off the gradients also save memory and computations

        with torch.no_grad():

            model.eval()

            print("testing.....")

            for images , labels in testload:

                #forward propagation

                test_sof_log = model(images)

                #apending test loss to represent total avg test loss

                test_loss += loss(test_sof_log,labels)

                #getting the class probabilities

                ps = torch.exp(test_sof_log)

                #getting the top class and its probability 

                top_p , top_class = ps.topk(1 , dim=1)

                #counting how many predicted labels matches true labels

                #first we have to match the shape of labels and top_class

                equals = top_class == labels.view(*top_class.shape)

                #as the numbers are 0 and 1, taking mean will give the accuracy

                #because the numerator will be sum of all ones

                #equals is of byte type tensor and mean dont work on them

                #so convert equals to float type tensor

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        #train the model

        model.train()

        

        #prepare loss list for train and test

        train_losses.append(runnning_loss/len(trainload))

        test_losses.append(test_loss/len(testload))

        

        #printing out the result

        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(train_losses[-1]),

              "Test Loss: {:.3f}.. ".format(test_losses[-1]),

              "Test Accuracy: {:.3f}".format(accuracy/len(testload)))