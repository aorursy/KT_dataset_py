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
import csv

train_data = []

with open("/kaggle/input/digit-recognizer/train.csv","r") as f:

    csvreader = csv.reader(f)

    for line in csvreader:

        train_data.append(line)
print(len(train))
print(train[0])
print(train[0][:10])
print(train[0][-1])
import numpy as np

img = np.array(train[1][1:], dtype=np.uint8)
img = img.reshape(28,28)
import matplotlib.pyplot as plt

plt.imshow(img)
#import pytorch

import torch

import torch.nn as nn
# This is the class (i.e. the blueprint) for our neural network

# ANN stands for Artifical Neural Network



class ANNModel(nn.Module):

    

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(ANNModel, self).__init__()

        # Linear function 1: 784 --> 100

        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity 1

        self.sig1 = nn.Sigmoid()

        

        # Linear function 2 (readout): 100 --> 10

        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    

    def forward(self, x):

        # Linear function 1

        out = self.fc1(x)

        # Non-linearity 1

        out = self.sig1(out)

        

        # Linear function 2 (readout)

        out = self.fc2(out)

        return out
# instantiate ANN

input_dim = 28*28

hidden_dim = 150 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.

output_dim = 10



# Create ANN object from our class

model = ANNModel(input_dim, hidden_dim, output_dim)



# Cross Entropy Loss -> this is how we calculate the error in our ANN's predictions

error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_features = [line[1:] for line in train[1:]]

train_features = np.array(train_features, dtype=np.float64)/255.0
train_target = [line[0]  for line in train[1:]]

train_target = np.array(train_target, dtype=np.int16)
train_features_tensor = torch.from_numpy(train_features).float()

train_target_tensor = torch.from_numpy(train_target).long()
train_dataset = torch.utils.data.TensorDataset(

    train_features_tensor, 

    train_target_tensor

)

batch_size = 64

train_loader = torch.utils.data.DataLoader(

    train_dataset, 

    batch_size = batch_size, 

    shuffle = True

)
from torch.autograd import Variable



# How many times to go through the entire dataset

num_epochs = 2



# ANN model training

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):



        train = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and cross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

for images, labels in train_loader:

    test = Variable(images.view(-1, 28*28))

    

    # Forward propagation

    outputs = model(test)



    # Get predictions from the maximum value

    predicted = torch.max(outputs.data, 1)[1]



    # Total number of labels

    total = len(labels)



    # Total correct predictions

    correct = (predicted == labels).sum()

    break



accuracy = 100 * correct / float(total)

print(accuracy)
test = []

with open("/kaggle/input/digit-recognizer/test.csv","r") as f:

    csvreader = csv.reader(f)

    for line in csvreader:

        test.append(line)
test[0][:10]
test_features = test[1:]

test_features = np.array(test_features, dtype=np.float64)/255.0



test_features_tensor = torch.from_numpy(test_features).float()

fake_labels = torch.from_numpy(np.random.randint(0,1, (test_features_tensor.shape[0], ) ))





test_dataset = torch.utils.data.TensorDataset(

    test_features_tensor,

    fake_labels

)



batch_size = 64

test_loader = torch.utils.data.DataLoader(

    test_dataset, 

    batch_size = batch_size, 

    shuffle = False

)
# TO DO: generate predictions

all_predictions = []

for images, fake_labels in test_loader:

    test_imgs = Variable(images.view(-1, 28*28))

    

    # Forward propagation

    outputs = model(test_imgs)



    # Get predictions from the maximum value

    predicted = torch.max(outputs.data, 1)[1]

#     print(predicted)

    all_predictions += predicted

print(len(all_predictions))
# TO DO: save predictions to CSV file

with open("submission.csv", "w") as f:

    f.write("ImageId,Label\n")

    for i, prediction in enumerate(all_predictions):

        line = f"{i+1},{prediction}\n"

        f.write(line)
import torch.nn.functional as F



class cnn(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # 28x28x1 -> 28x28x4

        self.cl1 = nn.Conv2d(1, 4, (3,3), padding=1)

        # 28x28x4 -> 14x14x4

        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # 14x14x4 -> 14x14x8

        self.cl2 = nn.Conv2d(4, 8, (3,3), padding=1)

        # 14x14x8 -> 7x7x8

        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.lin = nn.Linear(7*7*8, num_classes)

    

    def forward(self, x):

        out = F.relu(self.cl1(x))

        out = self.mp1(out)

        out = F.relu(self.cl2(out))

        out = self.mp2(out)

        out = self.lin(out.view(-1, 7*7*8))

        return out
model = cnn(num_classes = 10)
import csv

train_data = []

with open("/kaggle/input/digit-recognizer/train.csv","r") as f:

    csvreader = csv.reader(f)

    for line in csvreader:

        train_data.append(line)




train_features = [line[1:] for line in train_data[1:]]

train_features = np.array(train_features, dtype=np.float64)/255.0



train_features = train_features.reshape(42000, 1, 28,28)



train_target = [line[0]  for line in train_data[1:]]

train_target = np.array(train_target, dtype=np.int16)



train_features_tensor = torch.from_numpy(train_features).float()

train_target_tensor = torch.from_numpy(train_target).long()



train_dataset = torch.utils.data.TensorDataset(

    train_features_tensor, 

    train_target_tensor

)

batch_size = 64

train_loader = torch.utils.data.DataLoader(

    train_dataset, 

    batch_size = batch_size, 

    shuffle = True

)
for img, labs in train_loader:

    print(img.shape)

    break
error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
from torch.autograd import Variable



# How many times to go through the entire dataset

num_epochs = 2



# ANN model training

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):



        train = Variable(images)

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and cross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

test = []

with open("/kaggle/input/digit-recognizer/test.csv","r") as f:

    csvreader = csv.reader(f)

    for line in csvreader:

        test.append(line)



test_features = test[1:]

test_features = np.array(test_features, dtype=np.float64)/255.0



test_features = test_features.reshape(len(test)-1, 1, 28,28)



test_features_tensor = torch.from_numpy(test_features).float()

fake_labels = torch.from_numpy(np.random.randint(0,1, (test_features_tensor.shape[0], ) ))





test_dataset = torch.utils.data.TensorDataset(

    test_features_tensor,

    fake_labels

)



batch_size = 64

test_loader = torch.utils.data.DataLoader(

    test_dataset, 

    batch_size = batch_size, 

    shuffle = False

)
# TO DO: generate predictions

all_predictions = []

for images, fake_labels in test_loader:

    test_imgs = Variable(images)

    

    # Forward propagation

    outputs = model(test_imgs)



    # Get predictions from the maximum value

    predicted = torch.max(outputs.data, 1)[1]

#     print(predicted)

    all_predictions += predicted

print(len(all_predictions))
with open("submission.csv", "w") as f:

    f.write("ImageId,Label\n")

    for i, prediction in enumerate(all_predictions):

        line = f"{i+1},{prediction}\n"

        f.write(line)