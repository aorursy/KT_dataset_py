import numpy as np

import pandas as pd

import torch.nn.functional as F

import math

from torch.optim import lr_scheduler

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import metrics

import torch

import itertools

from torchvision import models

import torch.optim as optim

from matplotlib.ticker import MaxNLocator

import torchvision

import torchvision.transforms as transforms

from torch.autograd import Variable

from torch import nn

from torch.utils.data import Dataset, DataLoader

import os

from torch.nn import MaxPool2d

import chainer.links as L

from PIL import Image

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

plt.ion()
#This data is wrongly matched. Please execute this code to have the correct mapping of X and y values



data = np.load('../input/Sign-language-digits-dataset/X.npy')

#target = np.load('../input/Sign-language-digits-dataset/Y.npy') #unmatched with the data, not to use 

Y = np.zeros(data.shape[0])

Y[:204] = 9

Y[204:409] = 0

Y[409:615] = 7

Y[615:822] = 6

Y[822:1028] = 1

Y[1028:1236] = 8

Y[1236:1443] = 4

Y[1443:1649] = 3

Y[1649:1855] = 2

Y[1855:] = 5



X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size = .02, random_state = 2) ## splitting into train and test set
#Shapes of X_train and y_train

X_train.shape, y_train.shape
class DatasetProcessing(Dataset):

    

    #initialise the class variables - transform, data, target

    def __init__(self, data, target, transform=None): 

        self.transform = transform

        self.data = data.reshape((-1,64,64)).astype(np.float32)[:,:,:,None]

        # converting target to torch.LongTensor dtype

        self.target = torch.from_numpy(target).long() 

    

    #retrieve the X and y index value and return it

    def __getitem__(self, index): 

        return self.transform(self.data[index]), self.target[index]

    

    #returns the length of the data

    def __len__(self): 

        return len(list(self.data))
# preprocessing images and performing operations sequentially

# Firstly, data is converted to PILImage, Secondly, converted to Tensor

# Thirdly, data is Normalized

transform = transforms.Compose(

    [transforms.ToPILImage(), transforms.ToTensor()])





dset_train = DatasetProcessing(X_train, y_train, transform)





train_loader = torch.utils.data.DataLoader(dset_train, batch_size=4,

                                          shuffle=True, num_workers=4)
dset_test = DatasetProcessing(X_test, y_test, transform)

test_loader = torch.utils.data.DataLoader(dset_test, batch_size=4,

                                          shuffle=True, num_workers=4)
plt.figure(figsize = (16, 4))

for num, x in enumerate(X_train[0:6]):

    plt.subplot(1,6,num+1)

    plt.axis('off')

    plt.imshow(x)

    plt.title(y_train[num])
class Net(nn.Module):    

    

    # This constructor will initialize the model architecture

    def __init__(self):

        super(Net, self).__init__()

          

        self.cnn_layers = nn.Sequential(

            # Defining a 2D convolution layer

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            # Putting a 2D Batchnorm after CNN layer

            nn.BatchNorm2d(32),

            # Adding Relu Activation

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

          

        self.linear_layers = nn.Sequential(

            # Adding Dropout

            nn.Dropout(p = 0.5),

            nn.Linear(32 * 32 * 32, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 10),

        )

        

    # Defining the forward pass    

    def forward(self, x):

        

        # Forward Pass through the CNN Layers 

        x = self.cnn_layers(x)

        x = x.view(x.size(0), -1)

        # Forwrd pass through Fully Connected Layers

        x = self.linear_layers(x)

        return F.log_softmax(x) 
model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
########################################

#       Training the model             #

########################################

def train(epoch):

    model.train()

    exp_lr_scheduler.step()

    tr_loss = 0

    correct = 0

    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

            

        # Clearing the Gradients of the model parameters

        optimizer.zero_grad()

        output = model(data)

        pred = torch.max(output.data, 1)[1]

        correct += (pred == target).sum()

        total += len(data)

        

        # Computing the loss

        loss = criterion(output, target)

        

        # Computing the updated weights of all the model parameters

        loss.backward()

        optimizer.step()

        tr_loss = loss.item()

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {} %'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.item(),100 * correct / total))

            torch.save(model.state_dict(), './model.pth')

            torch.save(model.state_dict(), './optimizer.pth')

    train_loss.append(tr_loss / len(train_loader))

    train_accuracy.append(100 * correct / total)
########################################

#       Evaluating the model           #

########################################



def evaluate(data_loader):

    model.eval()

    loss = 0

    correct = 0

    total = 0

    for data, target in data_loader:

        data, target = Variable(data, volatile=True), Variable(target)

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = torch.max(output.data, 1)[1]

        total += len(data)

        correct += (pred == target).sum()

    loss /= len(data_loader.dataset)

    valid_loss.append(loss)    

    valid_accuracy.append(100 * correct / total)

    print('\nAverage Validation loss: {:.5f}\tAccuracy: {} %'.format(loss, 100 * correct / total))
n_epochs = 50

train_loss = []

train_accuracy = []

valid_loss = []

valid_accuracy = []

for epoch in range(n_epochs):

    train(epoch)

    evaluate(test_loader)
########################################

#       Plotting the Graph             #

########################################



def plot_graph(epochs):

    fig = plt.figure(figsize=(20,4))

    ax = fig.add_subplot(1, 2, 1)

    plt.title("Train - Validation Loss")

    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')

    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')

    

    ax = fig.add_subplot(1, 2, 2)

    plt.title("Train - Validation Accuracy")

    plt.plot(list(np.arange(epochs) + 1) , train_accuracy, label='train')

    plt.plot(list(np.arange(epochs) + 1), valid_accuracy, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('accuracy', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')
plot_graph(n_epochs)
_, (validation_data, target) = next(enumerate(test_loader))

with torch.no_grad():

    output = model(validation_data.cuda())

fig = plt.figure()

for i in range(4):

    plt.subplot(2,2,i+1)

    plt.tight_layout()

    plt.imshow(validation_data[i][0], interpolation='none')

    pred = output.data.max(1, keepdim=True)[1][i].item()

    plt.title("Prediction: {}".format(pred))
with torch.no_grad():

    output = model(validation_data.cuda())



softmax = torch.exp(output).cpu()

prob = list(softmax.numpy())



fig = plt.figure(figsize = (16, 8))

for i in range(0, 4):

    fig.tight_layout()

    plt.style.use('classic')

    plt.subplot(4,2, 2 * i + 1)

    plt.imshow(validation_data[i][0], interpolation='none')

    plt.xticks([])

    plt.yticks([])

    pred = output.data.max(1, keepdim=True)[1][i].item()

    plt.title("Prediction: {}".format(pred))

    plt.subplot(4,2, 2 * i + 2)

    plt.barh([0], [max(prob[i])])

    plt.yticks([])

    plt.title("Predicted Probability: {0:.2f}".format(max(prob[i])))
train_transform= transforms.Compose([

            transforms.ToPILImage(),

            transforms.RandomHorizontalFlip(), # Horizontal Flip

            transforms.RandomCrop(64, padding=2), # Centre Crop

            transforms.ToTensor(),  #Convereting the input to tensor

            ])

dset_train = DatasetProcessing(X_train, y_train, train_transform)

train_loader = torch.utils.data.DataLoader(dset_train, batch_size=4,

                                          shuffle=True, num_workers=4)
n_epochs = 50



train_loss = []

train_accuracy = []

valid_loss = []

valid_accuracy = []



for epoch in range(n_epochs):

    train(epoch)

    evaluate(test_loader)
# Plotting train and validation loss

plot_graph(n_epochs)
_, (validation_data, target) = next(enumerate(test_loader))

with torch.no_grad():

    output = model(validation_data.cuda())

fig = plt.figure()

for i in range(4):

    plt.subplot(2,2,i+1)

    plt.tight_layout()

    plt.imshow(validation_data[i][0], interpolation='none')

    pred = output.data.max(1, keepdim=True)[1][i].item()

    plt.title("Prediction: {}".format(pred))
with torch.no_grad():

    output = model(validation_data.cuda())



softmax = torch.exp(output).cpu()

prob = list(softmax.numpy())



fig = plt.figure(figsize = (16, 8))

for i in range(0, 4):

    fig.tight_layout()

    plt.style.use('classic')

    plt.subplot(4,2, 2 * i + 1)

    plt.imshow(validation_data[i][0], interpolation='none')

    plt.xticks([])

    plt.yticks([])

    pred = output.data.max(1, keepdim=True)[1][i].item()

    plt.title("Prediction: {}".format(pred))

    plt.subplot(4,2, 2 * i + 2)

    plt.barh([0], [max(prob[i])])

    plt.yticks([])

    plt.title("Predicted Probability: {0:.2f}".format(max(prob[i])))
#######################################################################

#       Defining various model architectures for ensembling           #

#######################################################################

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

    

        #1st net

        self.conv1_1 = nn.Conv2d(1, 4, kernel_size=(4, 4), padding = (3, 3), stride=(2, 2)) #34 * 34

        self.conv1_2 = nn.Conv2d(4, 8, kernel_size=(4, 4), padding = (2, 2), stride=(2, 2)) #18 * 18 #mp

        self.fc1_1 = nn.Linear(8 * 9 * 9, 32) #dropout = 0.2

        self.fc1_1_drop = nn.Dropout(p=0.2)

        self.fc1_2 = nn.Linear(32, 10)



        #2nd net

        self.conv2_1 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #64 * 64

        self.conv2_2 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #64 * 64 #mp

        self.conv2_2_drop = nn.Dropout2d()

        self.conv2_3 = nn.Conv2d(6, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #64 * 64

        self.conv2_4 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #64 * 64 #mp

        self.conv2_4_drop = nn.Dropout2d()

        self.fc2_1 = nn.Linear(24 * 16 * 16, 120) #dropout = 0.5

        self.fc2_1_drop = nn.Dropout(p=0.5)

        self.fc2_2 = nn.Linear(120, 10)



        #3rd net

        self.conv3_1 = nn.Conv2d(1, 4, kernel_size=(5, 5), stride=(3, 3), padding=(2, 2)) #22 * 22

        self.conv3_2 = nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #22 * 22 #mp

        self.conv3_3 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #11 * 11

        self.fc3_1 = nn.Linear(16 * 11 * 11, 64) #dropout = 0.4

        self.fc3_1_drop = nn.Dropout(p=0.4)

        self.fc3_2 = nn.Linear(64, 10)



    def forward(self, x, y, z):

        x = F.relu(self.conv1_1(x))

        x = F.relu(self.pool(self.conv1_2(x)))

        x = x.view(-1, 648) #can also do x.view(-1, 1)

        x = F.relu(self.fc1_1(x))

        x = F.dropout(x, training = self.training)

        x = F.relu(self.fc1_2(x))



        y = F.relu(self.conv2_1(y))

        y = F.relu(self.pool(self.conv2_2_drop(self.conv2_2(y))))

        y = F.relu(self.conv2_3(y))

        y = F.relu(self.pool(self.conv2_4_drop(self.conv2_4(y))))

        y = y.view(-1, 256 * 24)

        y = F.relu(self.fc2_1_drop(self.fc2_1(y)))

        y = F.relu(self.fc2_2(y))



        z = F.relu(self.conv3_1(z))

        z = F.relu(self.pool(self.conv3_2(z)))

        z = F.relu(self.conv3_3(z))

        z = z.view(-1, 16 * 121)

        z = F.relu(self.fc3_1_drop(self.fc3_1(z)))

        z = F.relu(self.fc3_2(z))



        x = torch.cat((x, y, z))



        return F.sigmoid(x)
########################################

#       Plotting the Graph             #

########################################



def plot_graphs(train_loss, valid_loss, epochs):

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20,4))

    ax = fig.add_subplot(1, 2, 1)

    plt.title("Train Loss")

    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('train_loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')

    ax = fig.add_subplot(1, 2, 2)

    plt.title("Validation Loss")

    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='test')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('vaidation _loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')
##################################################

#       Training the ensembled model             #

##################################################



def train_ensemble(epoch):

    model.train()

    exp_lr_scheduler.step()

    tr_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)

        #print(data.size())

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        optimizer.zero_grad()

        target = torch.cat((target, target, target))

        output  = model(data, data, data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        tr_loss += loss.item()

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.item()))

    train_loss.append(tr_loss / len(train_loader))
####################################################

#       Evaluating the ensembled model             #

####################################################





def evaluate_ensemble(data_loader):

    model.eval()

    loss = 0

    

    for data, target in data_loader:

        data, target = Variable(data, volatile=True), Variable(target)

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        target = torch.cat((target, target, target))

        output  = model(data, data, data)

        # Using the functional API

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]

        

    loss /= len(data_loader.dataset)

    valid_loss.append(loss)    

    print('\nAverage Validation loss: {:.5f}\n'.format(loss))
# Initializing the ensembled model

model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.001)



criterion = nn.CrossEntropyLoss()



exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()



n_epochs = 5



train_loss = []

valid_loss = []



# Training and Evaluating for n_epochs

for epoch in range(n_epochs):

    train_ensemble(epoch)

    evaluate_ensemble(test_loader)
plot_graphs(train_loss, valid_loss, n_epochs)