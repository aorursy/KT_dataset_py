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
import torch
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
train = pd.read_csv("../input/digit-recognizer/train.csv")
x_test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()

# split data into features(pixels) and labels(numbers from 0 to 9)
trainX_numpy = train.loc[:,train.columns != "label"].values/255 # normalization
trainY_numpy = train.label.values

# train test split. Size of train data is 80% and size of test data is 20%. 
trainX ,testX ,trainY ,testY = train_test_split(trainX_numpy,trainY_numpy,test_size = 0.2) 

trainX = torch.from_numpy(trainX)
trainY = torch.from_numpy(trainY).type(torch.LongTensor) # data type is long

trainX = trainX.view(-1,1,28,28)

# create feature and targets tensor for test set.
testX = torch.from_numpy(testX)
testY = torch.from_numpy(testY).type(torch.LongTensor) # data type is long

testX = testX.view(-1,1,28,28)


print(trainX.shape,trainY.shape,testX.shape,testY.shape)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# Pytorch train and test sets
train = TensorDataset(trainX,trainY)
test = TensorDataset(testX,testY)

# data loader
batch_size = 128
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
valid_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
fig = plt.figure(figsize = (8,8))
img = images[1]
img = img.squeeze(0)
width, height = img.shape
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(float(img[x][y]),2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                horizontalalignment='center',
                verticalalignment='center', size=8,
                color='white' if img[x][y]<thresh else 'black')

import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        #28*28*1
        self.conv1 = conv(1,16,kernel_size=3,stride=1,padding=1,batch_norm=True)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #14*14*16
        self.conv2 = conv(16,32,kernel_size=3,stride=1,padding=1,batch_norm=True)
        
        #7*7*64
        self.conv3 = conv(32,64,kernel_size=3,stride=1,padding=1,batch_norm=True)
        
        #3*3*128
        self.fc1 = nn.Linear(64*3*3,256)
        self.fc2 = nn.Linear(256,10)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64*3*3)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
import torch.optim as optim

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# number of epochs to train the model
n_epochs = 500 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.float())
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.float())
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_mnist.pt')
        valid_loss_min = valid_loss
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
PATH = './model_mnist.pt'
model.load_state_dict(torch.load(PATH))
model.eval()
BATCH_SIZE = 128
def evaluate(model):
    correct = 0
    
    for data,target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        output = model(data)
        predicted = torch.max(output,1)[1]
        correct += (predicted == target).sum()
    print("Test accuracy:{:.3f} ".format( float(correct) / (len(valid_loader)*BATCH_SIZE)))
evaluate(model)
sub_test_x = pd.read_csv("../input/digit-recognizer/test.csv")
sub_test_x.head()
# create feature and targets tensor for test set.
sub_test_x = sub_test_x.loc[:,:].values/255 # normalization
sub_test_x = torch.from_numpy(sub_test_x)
sub_test_x = sub_test_x.view(-1,1,28,28)
print(len(sub_test_x),sub_test_x.shape)
batch_size = 128
test_loader = DataLoader(sub_test_x, batch_size = batch_size, shuffle = False)
dataiter = iter(test_loader)
test_images = dataiter.next()
print(type(test_images))
print(test_images.shape)

plt.imshow(test_images[0].numpy().squeeze(), cmap='Greys_r');
def Y_predict(model):
    correct = 0
    sub_test_y = []
    for data in test_loader:
        if train_on_gpu:
            data = data.cuda()
        data = data.float()
        output = model(data)
        predicted = torch.max(output,1)[1]
        for value in predicted:
            cpu_value = value.cpu()
            cpu_value = cpu_value.numpy()
            sub_test_y.append(cpu_value.tolist())

    
    return sub_test_y
sub_test_y = Y_predict(model)


my_submission = pd.DataFrame({'ImageId': list(range(1,len(sub_test_y)+1)), 'Label':sub_test_y})
my_submission.to_csv('Final_submission1.csv', index=False)