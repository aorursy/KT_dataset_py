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
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
labels= glob.glob('/kaggle/input/visiolab/train/labels/*.txt')
y_train = []
new=[]
#Reading file and extracting paths and labels
for file in labels:
    with open(file, 'r') as new_file:
        infoFile =pd.read_csv(new_file) #Reading all the lines from File
        for line in infoFile:#Reading line-by-line
            words = line.split(" ")
            y_train.append(int(words[0]))
Y =np.array(y_train)
print(Y)
images = glob.glob('/kaggle/input/visiolab/train/images/*.jpg')
X_train=[]
for im in images:
    img= cv2.imread(im)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(int(100),int(100)))
    img = img/255
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    X_train.append(img)
X=np.array(X_train)
print(X.shape)
import matplotlib.pyplot as plt
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(X[i], cmap='gray')
plt.subplot(222), plt.imshow(X[i+25], cmap='gray')
plt.subplot(223), plt.imshow(X[i+50], cmap='gray')
plt.subplot(224), plt.imshow(X[i+75], cmap='gray')
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1,train_size=0.9, shuffle=True, stratify=Y)
print(X_train.shape)
X_train = X_train.reshape(209, 1,100,100)
X_train  = torch.tensor(X_train)

y_train= y_train.astype(int);
y_train= torch.tensor(y_train,dtype=torch.long)
X_val = X_val.reshape(24,1,100,100)
X_val  = torch.tensor(X_val)

y_val = y_val.astype(int);
y_val = torch.tensor(y_val, dtype=torch.long)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output
class SimpleNet(nn.Module):   
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(24,36 , kernel_size=3, stride=1, padding=1),
            nn.Conv2d(36,36 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(36 * 6* 6, 7)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
use_cuda =True


from torch.optim import Adam


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
model = SimpleNet()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()
# Create model, optimizer and loss function


if use_cuda and torch.cuda.is_available():
  model.cuda



#Define the optimizer and loss function



train_losses= []
val_losses = []
def train(epoch):
    model.train()
    tr_loss = 0

    train_acc = 0.0
    valid_acc= 0.0
    
    # getting the training set
    train_X, train_y = Variable(X_train), Variable(y_train)
    # getting the validation set
    val_X, val_y = Variable(X_val), Variable(y_val)

    # converting the data into GPU format
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(train_X)
    output_val = model(val_X)

    # computing the training and validation loss
    loss_train = loss_fn(output_train, train_y)
    loss_val = loss_fn(output_val, val_y)
    train_losses.append(loss_train)
    val_losses.append(loss_val)


    _, prediction = torch.max(output_train, 1)
            
    train_acc += torch.sum(prediction == train_y.data)
    
    _, prediction_valid = torch.max(output_val, 1)
             
    valid_acc += torch.sum(prediction_valid == val_y.data)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_train,'\t', 'loss_val :', loss_val)
        
  
n_epochs = 30
for epoch in range(n_epochs):
    train(epoch)
    


plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
from sklearn.metrics import accuracy_score
with torch.no_grad():
    output = model(X_train)
    
softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(y_train, predictions)
with torch.no_grad():
    output = model(X_train)
    
softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(y_train, predictions)
test_images = glob.glob('/kaggle/input/visiolab/test/images/*.jpg')
X_test=[]
for im in test_images:
    img= cv2.imread(im)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(int(100),int(100)))
    img = img/255
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    X_test.append(img)
X_test=np.array(X_test)
print(X_test.shape)
X_test = X_test.reshape(50, 1,100,100)
X_test  = torch.tensor(X_test)

with torch.no_grad():
    output = model(X_test)

softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
sample_submission =pd.read_csv('/kaggle/input/visiolab/sample_submission.csv')
sample_submission['Category'] = predictions
sample_submission.head()

sample_submission.to_csv('submission.csv', index=False)
