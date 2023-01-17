import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch



from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import time

import os

import glob

import shutil



device = "cuda" if torch.cuda.is_available() else "cpu"

total_training_iter = 10
transformation = transforms.Compose([transforms.ToTensor() , 

                                    transforms.Normalize((0.1307,) , (0.3801,))])



transformation2 = transforms.Compose([transforms.ToTensor()])
train_val_dataset_normalize = datasets.MNIST('data/' , train = True , transform=transformation , download=True)

test_dataset_normalize = datasets.MNIST('data/' , train = False , transform= transformation , download=False)
train_size = int(0.916666667 * len(train_val_dataset_normalize))        #55000 train size , 5000 validation

val_size = len(train_val_dataset_normalize) - train_size

train_dataset_normalize, val_dataset_normalize = torch.utils.data.random_split(train_val_dataset_normalize , [train_size, val_size])
print("Training size : " , len(train_dataset_normalize))

print("Test size : " , len(test_dataset_normalize))

print("Validation size : " , len(val_dataset_normalize))
train_val_dataset_unnormalize = datasets.MNIST('data/' , train = True , transform=transformation2 , download=True)

test_dataset_unnormalize = datasets.MNIST('data/' , train = False , transform= transformation2 , download=False)
train_size = int(0.916666667 * len(train_val_dataset_unnormalize))        #55000 train size , 5000 validation

val_size = len(train_val_dataset_unnormalize) - train_size

train_dataset_unnormalize, val_dataset_unnormalize = torch.utils.data.random_split(train_val_dataset_unnormalize , [train_size, val_size])
def get_loaders (train , validation , test):

    train_loader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True)

    test_loader = torch.utils.data.DataLoader(test , batch_size = 32 , shuffle = True)

    val_loader = torch.utils.data.DataLoader(validation , batch_size = 32 , shuffle = True)

    

    return (train_loader, val_loader , test_loader)
train_loader ,  val_loader , test_loader = get_loaders(train_dataset_normalize 

                                                       , val_dataset_normalize , test_dataset_normalize)
# building the architecture from scratch

class Net(nn.Module):

    def __init__ (self , hidden_layers ,activation):

        super().__init__()             #this line will ensure that the class Net is the child class of nn.Module

                                        # all the methods available for nn.Module will also be available for Net

                                        #this just passes the parameters of Net to nn.Module                

        self.act = nn.Sigmoid()

        if (activation == 'sig'):

            self.act = nn.Sigmoid()

        elif (activation == 'tanh'):

            self.act = nn.Tanh()

        elif (activation == 'relu'):

            self.act = nn.ReLU()

        elif (activation == 'leakyRelu'):

            self.act = nn.LeakyReLU()

        

        self.fc1 = nn.Linear(784 , hidden_layers)                             

        self.fc2 = nn.Linear(hidden_layers , 10)



    def forward(self , x):

        x = x.view(-1, 784)

        x = self.act(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x)



        
# building the architecture from scratch

class Net2(nn.Module):

    def __init__ (self , hidden_layers , activation):

        super().__init__()             #this line will ensure that the class Net is the child class of nn.Module

                                        # all the methods available for nn.Module will also be available for Net

                                        #this just passes the parameters of Net to nn.Module

        self.act = nn.Sigmoid()

        if (activation == 'sig'):

            self.act = nn.Sigmoid()

        elif (activation == 'tanh'):

            self.act = nn.Tanh()

        elif (activation == 'relu'):

            self.act = nn.ReLU()

        elif (activation == 'leakyRelu'):

            self.act = nn.LeakyReLU()

            

        self.fc1 = nn.Linear(784 , hidden_layers)

        self.fc2 = nn.Linear(hidden_layers , hidden_layers)

        self.fc3 = nn.Linear(hidden_layers , 10)



    def forward(self , x):

        x = x.view(-1, 784)

        x = self.act(self.fc1(x))

        x = self.act(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x)
# building the architecture from scratch

class Net3(nn.Module):

    def __init__ (self , hidden_layers , activation):

        super().__init__()             #this line will ensure that the class Net is the child class of nn.Module

                                        # all the methods available for nn.Module will also be available for Net

                                        #this just passes the parameters of Net to nn.Module

        self.act = nn.Sigmoid()

        if (activation == 'sig'):

            self.act = nn.Sigmoid()

        elif (activation == 'tanh'):

            self.act = nn.Tanh()

        elif (activation == 'relu'):

            self.act = nn.ReLU()

        elif (activation == 'leakyRelu'):

            self.act = nn.LeakyReLU()

        

        self.fc1 = nn.Linear(784 , hidden_layers)

        self.fc2 = nn.Linear(hidden_layers , hidden_layers)

        self.fc3 = nn.Linear(hidden_layers , hidden_layers)

        self.fc4 = nn.Linear(hidden_layers , 10)



    def forward(self , x):

        x = x.view(-1, 784)

        x = self.act(self.fc1(x))

        x = self.act(self.fc2(x))

        x = self.act(self.fc3(x))

        x = self.fc4(x)

        return F.log_softmax(x)
# **training phase**

# dropout removes a percentage of values, which should not happen in the validation or testing phase

# For training mode, we  calculate the gradients and change the model parameters, 

# but backpropagation is not required during the testing or validation phase





def fit(epoch , model , data_loader , phase = 'training' , volatile = False):

    if phase == 'training':

        model.train(True)

    if phase == 'validation':

        model.eval()

        volatile = True

        

    running_loss = 0.0

    running_correct = 0

    

    #this will load each batch

    #batch_idx is the index of the batch

    for batch_idx , (data, target) in enumerate(data_loader):

        data, target = data.to(device) , target.to(device) 

        

        

        if phase == 'training':

            optimizer.zero_grad()

            

        output = model(data)

        loss = F.nll_loss(output , target)

        _ , preds = torch.max(output.data , 1) 

        

        running_loss += F.nll_loss (output , target , size_average = False).item()

        running_correct += torch.sum(target == preds)

        

        if phase == 'training':

            loss.backward()

            optimizer.step()

                

    loss = running_loss / len(data_loader.dataset)

    accuracy = (100.0 * running_correct.item()) / len(data_loader.dataset)

    print( "Epoch : " ,epoch, " ",phase ,  "  Loss : " , loss , " accuracy : " ,accuracy)  

    return loss , accuracy

def do_training (model , optimizer , epochs , train_loader , val_loader):



    train_losses = []

    train_accuracy = []



    val_loss = []

    val_accuracy = []



    for epoch in range(1,epochs+1):

        epoch_loss , epoch_accuracy = fit(epoch , model , train_loader , phase = 'training' )

        val_epoch_loss , val_epoch_accuracy = fit(epoch , model , val_loader , phase = 'validation')



        train_losses.append(epoch_loss)

        train_accuracy.append(epoch_accuracy)

        val_loss.append(val_epoch_loss)

        val_accuracy.append(val_epoch_accuracy)

    

    #plotting the training and validation loss

    plt.plot(range(1 , len(train_losses) + 1) , train_losses , 'bo' , label = 'training_loss' )

    plt.plot(range(1 , len(val_loss)+ 1) , val_loss, 'r' , label = 'validation_loss')

    plt.legend()

    plt.show()
#define the loaders

train_loader ,  val_loader , test_loader = get_loaders(train_dataset_normalize 

                                                       , val_dataset_normalize , test_dataset_normalize)
%%time

model = Net(hidden_layers=32 , activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
#define the loaders

train_loader ,  val_loader , test_loader = get_loaders(train_dataset_unnormalize 

                                                       , val_dataset_unnormalize , test_dataset_unnormalize)
%%time

model = Net(hidden_layers=32, activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
#define the loaders

train_loader ,  val_loader , test_loader = get_loaders(train_dataset_normalize 

                                                       , val_dataset_normalize , test_dataset_normalize)
%%time

model = Net2(hidden_layers=32 , activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1, model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net3(hidden_layers=32 , activation= 'sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=32 , activation = 'sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.001)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=32 , activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.0001)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=64 , activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter, train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=128 , activation='sig').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=128 , activation='relu').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=128 , activation='leakyRelu').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
%%time

model = Net2(hidden_layers=128 , activation='tanh').to(device)

optimizer = optim.SGD(model.parameters() , lr = 0.1)

do_training(model , optimizer , total_training_iter , train_loader , val_loader)
test_loss , test_accuracy = fit(1 , model , test_loader , phase = 'validation' )

print("Test Loss : " , test_loss)

print("Test accuracy {} ".format(test_accuracy))
#memory allocated in bytes on cuda

torch.cuda.max_memory_allocated()