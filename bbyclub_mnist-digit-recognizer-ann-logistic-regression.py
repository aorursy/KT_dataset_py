# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Prepare Dataset
# load data
train = pd.read_csv(r"../input/digit-recognizer/train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()
class LogisticRegressionModel(nn.Module):
    def __init__ (self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__() #Inherit the Neural Network Model
        self.linear= nn.Linear(input_dim, output_dim)
    def forward(self,x): #forward metodu pytorch tarafından otomatik olarak çağrılır.
        out=self.linear(x)
        return out

input_dim= 28*28
output_dim= 10

model= LogisticRegressionModel(input_dim, output_dim)

error= nn.CrossEntropyLoss()

learning_rate=0.001
optimizer= torch.optim.SGD(model.parameters(), learning_rate)
#Training the Model
count=0
loss_list=[]
iteration_list=[]
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): #data loader bana iki tane değer return ediyor. Tuple ve index sayısı
        #tuple içinde resimler ve karşılık gelen labellar var.
        
        #Define variables
        train= Variable(images.view(-1,28*28)) #gradient hesabı yapıyorum ve bu gradientleri tutması için variable lara ihtiyaç var.        
        #view metodu numpy daki reshape metoduna karşılık geliyor.
        labels=Variable(labels)
        
        #Clear gradients
        optimizer.zero_grad()
        
        #Forward propagation
        outputs=model(train) #modeli çağırdığımda forward metodunu da otomatik olarak çağırmış oluyorum.
        
        #calculate softmax and cross entropy loss Softmax fonksiyonu bunun içerisinde
        loss=error(outputs, labels)
        
        #calculate gradients
        loss.backward()
      
        #update parameters
        optimizer.step() #modelin parametreleri optimizer ın içerisinde. 
        
        count += 1
        #prediction #50 adımda bir test ediyorum.
        if count%50 ==0:
            #calculate acc
            correct=0
            total=0
            #predict test dataset
            for images, labels in test_loader:
                test= Variable(images.view(-1,28*28))
                
                #Forward propagation
                outputs= model(test)
                
                #get predictions from the maximum value
                predicted= torch.max(outputs.data,1)[1]
                
                #Total number of labels
                total+=len(labels)
                
                #Total correction predictions
                correct+= (predicted==labels).sum()
                
            accuracy=100 * correct / float(total)
            # store loss and iteration 
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count%500 ==0:
            #Print loss
            print("Iteration: {} Loss: {} Accuracy: {}%".format(count, loss.data, accuracy))
#visualization
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")
plt.show()
# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
# Create ANN Model
class ANNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        
        # Linear function 1: 784 --> 150
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 150 --> 150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()
        
        # Linear function 3: 150 --> 150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.elu3 = nn.ELU()
        
        # Linear function 4 (readout): 150 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)
        
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

# instantiate ANN
input_dim = 28*28
hidden_dim = 150 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
output_dim = 10

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(type(outputs))
print(type(labels))
print(type(featuresTrain))
print(type(targetsTrain))
print(type(featuresTest))
print(type(targetsTest))
print(type(train))
print(type(test))
print(type(labels))

# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader:

                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
