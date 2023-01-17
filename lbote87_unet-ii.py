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

            pos = np.random.randint(0, high=(len(dictt['pleth'][0])-(20*self.num_samples)))

             

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



trainset = SelectRecord(path_train, num_samples, True)

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

        self.conv3 = nn.Conv1d(64, 128, 3, stride=1, padding=1) # "same" convolution

        self.conv4 = nn.Conv1d(128, 256, 3, stride=1, padding=1) # "same" convolution

        self.conv5 = nn.Conv1d(256, 512, 3, stride=1, padding=1) # "same" convolution

        self.conv6 = nn.Conv1d(512, 1024, 3, stride=1, padding=1) # "same" convolution

        self.conv7 = nn.Conv1d(1024, 2048, 3, stride=1, padding=1) # "same" convolution

        self.conv8 = nn.Conv1d(2048, 1024, 3, stride=1, padding=1) # "same" convolution

        self.conv9 = nn.Conv1d(1024+1024, 512, 3, stride=1, padding=1) # "same" convolution

        self.conv10 = nn.Conv1d(512+512, 256, 3, stride=1, padding=1) # "same" convolution

        self.conv11 = nn.Conv1d(256+256, 128, 3, stride=1, padding=1) # "same" convolution

        self.conv12 = nn.Conv1d(128+128, 64, 3, stride=1, padding=1) # "same" convolution

        self.conv13 = nn.Conv1d(64+64, 32, 3, stride=1, padding=1) # "same" convolution

        self.conv14 = nn.Conv1d(32+32, 1, 3, stride=1, padding=1) # "same" convolution

        

        ## Batch normalization layers

        self.bn1 = nn.BatchNorm1d(32)

        self.bn2 = nn.BatchNorm1d(64)

        self.bn3 = nn.BatchNorm1d(128)

        self.bn4 = nn.BatchNorm1d(256)

        self.bn5 = nn.BatchNorm1d(512)

        self.bn6 = nn.BatchNorm1d(1024)

        self.bn7 = nn.BatchNorm1d(2048)

        self.bn8 = nn.BatchNorm1d(1024)

        self.bn9 = nn.BatchNorm1d(512)

        self.bn10 = nn.BatchNorm1d(256)

        self.bn11 = nn.BatchNorm1d(128)

        self.bn12 = nn.BatchNorm1d(64)

        self.bn13 = nn.BatchNorm1d(32)

    

        ## Dropout layer with drop probability

        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x):

    

        x1 = self.conv1(x)

        x1 = self.bn1(x1)

        x1 = F.relu(x1)

        x1 = self.dropout(x1)

    

        x2 = self.conv2(x1)

        x2 = self.bn2(x2)

        x2 = F.relu(x2)

        x2 = self.dropout(x2)

    

        x3 = self.conv3(x2)

        x3 = self.bn3(x3)

        x3 = F.relu(x3)

        x3 = self.dropout(x3)

    

        x4 = self.conv4(x3)

        x4 = self.bn4(x4)

        x4 = F.relu(x4)

        x4 = self.dropout(x4)

    

        x5 = self.conv5(x4)

        x5 = self.bn5(x5)

        x5 = F.relu(x5)

        x5 = self.dropout(x5)

    

        x6 = self.conv6(x5)

        x6 = self.bn6(x6)

        x6 = F.relu(x6)

        x6 = self.dropout(x6)

        

        x7 = self.conv7(x6)

        x7 = self.bn7(x7)

        x7 = F.relu(x7)

        x7 = self.dropout(x7)

        

        x8 = self.conv8(x7)

        x8 = self.bn8(x8)

        x8 = F.relu(x8)

        x8 = self.dropout(x8)

        

        x9 = torch.cat([x6, x8], 1)

        x9 = self.conv9(x9)

        x9 = self.bn9(x9)

        x9 = F.relu(x9)

        x9 = self.dropout(x9)

    

        x10 = torch.cat([x5, x9], 1)

        x10 = self.conv10(x10)

        x10 = self.bn10(x10)

        x10 = F.relu(x10)

        x10 = self.dropout(x10)

     

        x11 = torch.cat([x4, x10], 1)

        x11 = self.conv11(x11)

        x11 = self.bn11(x11)

        x11 = F.relu(x11)

        x11 = self.dropout(x11)

        

        x12 = torch.cat([x3, x11], 1)

        x12 = self.conv12(x12)

        x12 = self.bn12(x12)

        x12 = F.relu(x12)

        x12 = self.dropout(x12)

        

        x13 = torch.cat([x2, x12], 1)

        x13 = self.conv13(x13)

        x13 = self.bn13(x13)

        x13 = F.relu(x13)

        x13 = self.dropout(x13)

    

        ## Output tensor

        x14 = torch.cat([x1, x13], 1)

        x14 = self.conv14(x14)



        return x14
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