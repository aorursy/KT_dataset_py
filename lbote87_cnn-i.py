# Import necessary libraries



import numpy as np

import os

import pickle

import wfdb

import torch

from torch import nn

import torch.nn.functional as F

from torchvision import datasets

import time

import matplotlib.pyplot as plt

%matplotlib inline
# Device for training -> CPU or GPU



if torch.cuda.is_available():

    device = torch.device("cuda:"+str(torch.cuda.current_device()))

    print("GPU available")

    print("Device:",torch.cuda.current_device())

    print("Model:",torch.cuda.get_device_name(torch.cuda.current_device()))

else:

    device = torch.device("cpu")

    print("GPU doesn't available")
# Set NumPy and PyTorch seeds for reproducibility



seed = 1234



## Set Numpy seed

np.random.seed(seed)



## Set Pytorch seed

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)
# Path



path_train = "../input/trainset/trainset"

path_val = "../input/valset/valset"

path_test = "../input/testset/testset"
# Records info



train_records = os.listdir(path_train)

train_records.sort() # inplace

print('Number of train records: ' + str(len(train_records)))



val_records = os.listdir(path_val)

val_records.sort() # inplace

print('Number of validation records: ' + str(len(val_records)))



test_records = os.listdir(path_test)

test_records.sort() # inplace

print('Number of test records: ' + str(len(test_records)))
# Define the class to select randomly a record (ABP and PLETH signals)



class SelectRecord(torch.utils.data.Dataset):

    

    def __init__(self, path, num_samples=500, random=True):

        self.path = path

        self.num_samples = num_samples # number of samples per signal

        self.random = random

        self.list_files = os.listdir(path)

        self.list_files.sort()

        

    ## Override to give PyTorch size of dataset

    def __len__(self):

        return len(self.list_files)

                                                  

    ## Override to give PyTorch access to any image on the dataset

    def __getitem__(self, index):

        

        with open(self.path + '/' + self.list_files[index],'rb') as file:

            dictt = pickle.load(file)

        

        X_ = dictt['pleth'][0] # numpy array

        Y_ = dictt['abp'][0] # numpy array

        

        if self.random:

            pos = np.random.randint(0, high=(len(dict['pleth'][0])-(20*self.num_samples)))

             

        else:

            pos = 0

            

        while True:

            X = X_[pos:pos+self.num_samples]

            Y = Y_[pos:pos+self.num_samples]

            if not(np.any(np.isnan(X))) and not(np.any(np.isnan(Y))):

                break

            else:

                pos += int(self.num_samples/3)

                

        return (torch.from_numpy(X).type(torch.FloatTensor).unsqueeze_(0), torch.from_numpy(Y).type(torch.FloatTensor).unsqueeze_(0)) # return float tensor
# Define variables



num_samples = 150 # number of samples per signal

batch_size = 256

epochs = 100
# Load the training and validation dataset



trainset = SelectRecord(path_train, num_samples, False)

valset = SelectRecord(path_val, num_samples, False)

testset = SelectRecord(path_test, num_samples, False)
k = 0
# Display (run several times this cell to display different signals)



print(k)



## Training set

signal_train, targets_train = trainset[k]



plt.figure(1)

plt.plot(signal_train[0,:].numpy(), 'b', label='PLETH')

plt.legend(loc='upper right')

plt.show()



plt.figure(2)

plt.plot(targets_train[0,:].numpy(), 'g', label='ABP')

plt.legend(loc='upper right')

plt.show()



## Validation set

signal_val, targets_val = valset[k]



plt.figure(1)

plt.plot(signal_val[0,:].numpy(), 'b', label='PLETH')

plt.legend(loc='upper right')

plt.show()



plt.figure(2)

plt.plot(targets_val[0,:].numpy(), 'g', label='ABP')

plt.legend(loc='upper right')

plt.show()



## Test set

signal_test, targets_test = testset[k]



plt.figure(1)

plt.plot(signal_test[0,:].numpy(), 'b', label='PLETH')

plt.legend(loc='upper right')

plt.show()



plt.figure(2)

plt.plot(targets_test[0,:].numpy(), 'g', label='ABP')

plt.legend(loc='upper right')

plt.show()



k += 1
# Create the dataset loader



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
# Define the Convolutional Neural Network



class Network(nn.Module):

    def __init__(self):

        super().__init__()



        ## Hidden layers

        self.conv1 = nn.Conv1d(1, 32, 3, stride=1, padding=1) # "same" convolution

        self.conv2 = nn.Conv1d(32, 64, 3, stride=1, padding=1) # "same" convolution

        self.conv3 = nn.Conv1d(64, 32, 3, stride=1, padding=1) # "same" convolution

        self.conv4 = nn.Conv1d(32, 1, 3, stride=1, padding=1) # "same" convolution

    

        ## Dropout layer with drop probability

        self.dropout = nn.Dropout(p=0.4)

    

    def forward(self, x):

    

        x = self.conv1(x)

        x = F.relu(x)

        x = self.dropout(x)

    

        x = self.conv2(x)

        x = F.relu(x)

        x = self.dropout(x)

    

        x = self.conv3(x)

        x = F.relu(x)

        x = self.dropout(x)

    

        ## Output tensor

        x = self.conv4(x)



        return x
# Define the training process



def train_model(model, criterion, optimizer, traindataloader, valdataloader, epochs=5):



    ## Initialize variables

    training_time = 0

    validation_time = 0

    losses = []

    losses_val = []

  

    ## Set model to train mode

    model.train()



    for epoch in range(epochs):

    

        ## Print current epoch

        print('-' * 10)

        print('Epoch {}/{}'.format(epoch, epochs - 1))



        ## Initialize variables

        loss_epoch = 0

        loss_val_epoch = 0

    

        ## Start training time

        since0 = time.time()



        for signals, targets in traindataloader:    

      

            ## Send input tensors to device 

            signals = signals.to(device)

            targets = targets.to(device)



            ## Clear the gradients, do this because gradients are accumulated

            optimizer.zero_grad()



            ## Forward pass

            output = model(signals)



            ## Calculate the loss

            loss = criterion(output, targets)



            ## Backward pass

            loss.backward()



            ## Update weights

            optimizer.step()



            ## Accumulate loss of each batch of the epoch

            loss_epoch += loss.item()

            

            print('#', end = '')



        ## Append mean loss in each epoch

        losses.append(loss_epoch/len(traindataloader)) # len(traindataloader) = number of batches in traindataloader

    

        ## Print mean loss in each epoch

        print('Loss: {}'.format(loss_epoch/len(traindataloader))) # len(traindataloader) = number of batches in traindataloader

    

        ## Print training time in each epoch

        time_epoch = time.time() - since0

        print('Time: {:.0f}m {:.1f}s'.format(time_epoch // 60, time_epoch % 60))



        ## Accumulate training epoch time

        training_time += time_epoch



    ########################################## Validation ####################################

    

        ## Start validation time

        since1 = time.time()



        ## Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

      

            ## Set model to evaluation mode (dropout probability is 0 and BatchNorm)

            model.eval()



            for signals_val, targets_val in valdataloader:

                signals_val = signals_val.to(device)

                targets_val = targets_val.to(device)

                output_val = model(signals_val)

                loss_val = criterion(output_val, targets_val)

                loss_val_epoch += loss_val.item()

                print('#', end = '')



        ## Append mean loss in each epoch

        losses_val.append(loss_val_epoch/len(valdataloader)) # len(valdataloader) = number of batches in valdataloader

    

        ## Print mean loss in each epoch

        print('Validation loss: {}'.format(loss_val_epoch/len(valdataloader))) # len(valdataloader) = number of batches in valdataloader



        ## Print training time the epoch

        time_val_epoch = time.time() - since1

        print('Validation time: {:.0f}m {:.1f}s'.format(time_val_epoch // 60, time_val_epoch % 60))



        ## Accumulate validation epoch time

        validation_time += time_val_epoch



        ## Set model back to train mode

        model.train()

    ##########################################################################################



    ## Transform lists in NumPy arrays

    losses = np.array(losses)

    losses_val = np.array(losses_val)

  

    ## Print elapsed training time in all the epochs

    print('-' * 20)

    print('Total training time: {:.0f}m {:.1f}s'.format(training_time // 60, training_time % 60))

    print('Total validation time: {:.0f}m {:.1f}s'.format(validation_time // 60, validation_time % 60))

    print('Total  time: {:.0f}m {:.1f}s'.format((training_time + validation_time) // 60, (training_time + validation_time) % 60))

  

    ## Display training process (losses)

    plt.plot(losses, 'b', label='Training loss')

    plt.plot(losses_val, 'g', label='Validation loss')

    plt.legend(loc='upper right')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.show()

  

    return model
# Execution



## Create the model    

model = Network()

model = model.to(device) 



## Define loss function

criterion = nn.MSELoss()



## Define optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)



## Train the model

model = train_model(model, criterion, optimizer, trainloader, valloader, epochs)
# Prediction



## Send model to CPU for prediction

model = model.to("cpu")



## Set model to evaluation mode

model.eval()



## Create the dataset loader

testloader = torch.utils.data.DataLoader(testset, batch_size=len(test_records), shuffle=False)



signals_test, targets_test = next(iter(testloader))





## Send input tensors to device 

signals_test = signals_test.to("cpu")

targets_test = targets_test.to("cpu")



## Turn off gradients to speed up this part

with torch.no_grad():

    prediction = model(signals_test)



## Accuracy

mse = criterion(prediction, targets_test)

print(mse)
# Display 



i = np.random.randint(0, high=len(test_records))



print(i)



plt.figure(1)

plt.plot(signals_test[i,0,:].numpy(), 'b', label='PLETH')

plt.legend(loc='upper right')

plt.show()





plt.figure(2)

plt.plot(targets_test[i,0,:].numpy(), 'g', label='ABP')

plt.plot(prediction[i,0,:].numpy(), 'r', label='Predicted ABP')

plt.legend(loc='upper right')

plt.show()



i += 1