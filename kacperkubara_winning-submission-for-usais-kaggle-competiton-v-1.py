#Importing libraries:

import torch 

import torch.nn as nn

import pandas as pd

import numpy as np

import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn.functional as F



#Defining Net class based on nn.Module superclass

class Net(nn.Module):

    #Constructor of the Net class: 

    def __init__(self,inputsize,hiddensize,outputsize):

        super(Net, self).__init__()

        #Initializes two linear functions based on the constructor parameters

        self.fc1=nn.Linear(inputsize,hiddensize)

        self.fc2=nn.Linear(hiddensize,outputsize)

        

        #fc1 transforms m x inputsize matrix into m x hiddensize 

        #fc2 transforms m x hiddensize matrix into m x outputsize

    

    #Forward propagation function:

    def forward(self,x):

        out = self.fc1(x)

        out = self.fc2(out)



        return out
#Defining the train function:

def trainNN(model,input,target,alpha,maxit,Xtest,Ytest):

    

    #Using GPU to do calculations:

    input = input.cuda()

    target = target.cuda()

    Xtest=Xtest.cuda()

    Ytest=Ytest.cuda()

    

    

    #Zeroing values of cost histories

    costHistory=np.zeros(maxit);

    testHistory=np.zeros(maxit);



    

    for i in range (0,maxit):



        #Chossing the optimizer and resetting the gradient from previous iteration

        optimizer = optim.SGD(model.parameters(), lr=alpha)

        optimizer.zero_grad()

        

        #Calculating lose based on MSE error between target and output of network

        loss = nn.MSELoss()

        output = model.forward(input)

        cost=loss(output, target)

        

        #Adding the costs values to the history variables

        costHistory[i]=cost;

        testHistory[i]=loss(model.forward(Xtest),Ytest)

        

        #Performing a gradient decend step

        cost.backward()

        optimizer.step()

        

        #Print progress notification

        if((i/maxit*100)%5==0):

            print("Training completed in ",i/maxit*100," %.")

            

        #If cost is increasing than divide learning rate by 3    

        if(i>0):

            if(costHistory[i-1]<costHistory[i]):

                alpha=alpha/3

                print("Decreasing learning rate to",alpha)

        

        

    #Return the information about changes in Cost History for further analysis    

    return costHistory,testHistory
#---------------------------------------------Training the model ------------------------------------------------------------
#Load the train set data. Train.csv must be located in the same folder as this file.

train=pd.read_csv("train.csv")

#Shuffle the data set

train=train.sample(frac=1)

#Convert data to numpy object from pandas object

train=np.array(train)



#Create a train set X,Y out of 75% of data

#Columns 0 and 1 are discarded

X=train[0:12000,3:]

Y=train[0:12000,2]

#Ensure X and Y are matrix objects 

Y=np.matrix(Y,dtype=np.float32)

X=np.matrix(X,dtype=np.float32)

#Changing X and Y to PyTorch objects from numpy objects

X=torch.tensor(X)

Y=torch.tensor(Y)

#Ensure that Y is treated as a 2D array with dimension size of 1 in second axis and not as a 1D vector

Y=Y.view(-1,1)



#Create a test set Xtest,Ytest out of 25% of data

#Columns 0 and 1 are discarded

Xtest=train[12000:,3:]

Ytest=train[12000:,2]

#Ensure Xtest and Ytest are matrix objects 

Xtest=np.matrix(Xtest,dtype=np.float32)

Ytest=np.matrix(Ytest,dtype=np.float32)

#Changing Xtest and Ytest to PyTorch objects from numpy objects

Xtest=torch.tensor(Xtest)

Ytest=torch.tensor(Ytest)

#Ensure that Ytest is treated as a 2D array with dimension size of 1 in second axis and not as a 1D vector

Ytest=Ytest.view(-1,1)



#Perform feature normalization

X=F.normalize(X,dim=0)

Xtest=F.normalize(Xtest,dim=0)
#WARNING: This cell resets the model and will undo all the learning



#Creating network object of Net class



#The best size for hidden layes was found to be 1 and that is the value set as default in the code.

#To change the size of hidden layer change the second input parameter to Net constructor.

network=Net(18,1,1) 



#Using GPU to perfmorm operations on network

network.cuda()

#Reseting cost Hisotry for both train and test set

costHistory=None

testHistory=None

#Run learning function and save the returned cost history variables



#The best initial learning rate was found to be 0.0000001

#The best reults of learning on the test set occur after around 4800 iterations

#After that model starts to overfit

(costHistory_more,testHistory_more)=trainNN(network,X,Y,0.0000001,4800,Xtest,Ytest)



#Merge the returned cost history variables and global cost history variables

costHistory=np.append(costHistory,costHistory_more)

testHistory=np.append(testHistory,testHistory_more)
#Visulaize cost Hisotry over iterations

plt.plot(costHistory)

plt.plot(testHistory)
#This cell finds the iteration for which the cost was lowest on test set

#It also prints that cost

for i in range (0,np.size(testHistory)):

    if np.amin(testHistory[1:])==testHistory[i]:

        print(i)

print(np.amin(testHistory[1:]))
#This cells prints the cost of currently loaded model for test set

input=network.forward(Xtest.cuda())

output=Ytest.cuda()

loss=nn.MSELoss()

CVcost=loss(input,output)

print(CVcost)
#--------------------------------Making Predictions--------------------------------------------------------------------------
#Load the data for the submision test set (following the same procedure as beofre)

input=pd.read_csv("test.csv")

input=np.array(input)

X_input=input[:,2:]

X_input=np.matrix(X_input,dtype=np.float32)

X_input=torch.tensor(X_input)

X_input=X_input.cuda()

X_input=F.normalize(X_input,dim=0)
#Predict prizes and save to a csv file



(m,n)=np.shape(X_input)



#Predict Y based on X_input

h=network.forward(X_input)



#Change hypotesis h to numpy object

h=h.cpu().detach().numpy()



#Create the id table needed for submission

id=np.zeros((m,1))

for i in range(0,m):

    id[i][0]=i+45129

#Ensure the right data type for id table and hypotesis (int 32)    

id=np.matrix(id,dtype=np.int32)

h=np.matrix(h,dtype=np.int32)



#Join hypotesis and id table together

h=np.concatenate((h,id),axis=1)



#Convert result to pandas DataFrame

result=pd.DataFrame(h,columns=['price','id'])



#Save pandas data frame as sub.csv

result.to_csv('sub.csv',index = False)