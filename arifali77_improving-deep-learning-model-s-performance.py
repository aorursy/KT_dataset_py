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
from tqdm import tqdm # tqdm is a progress bar library with good support for nested loops and jupyter notebooks



# for reading and displaying image

from skimage.io import imread

from skimage.transform import resize

import matplotlib.pyplot as plt

%matplotlib inline
# for creating validation set

from sklearn.model_selection import train_test_split



# for evaluating the model 

from sklearn.metrics import accuracy_score



# Pytorch libraries and modules 

import torch

from torch.autograd import Variable

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from torch.optim import Adam, SGD



# torchvision for pre trained models

from torchvision import models
# loading dataset



train = pd.read_csv("/kaggle/input/emergency_vs_non-emergency_dataset/emergency_train.csv")

train.head()
# loading training images

train_img = []

for img_name in tqdm(train['image_names']):

    #print(img_name)

    

    # defining the image path

    image_path='/kaggle/input/emergency_vs_non-emergency_dataset/images/' + img_name

    #print(image_path)

    

    # reading the image

    img = imread(image_path)

    #print(img)

    

    # normalizing the pixel values 

    img = img/255

    #print(img)

    

    # resizing the image to (224,224,3)

    img = resize(img, output_shape=(224,224,3), mode='constant', anti_aliasing=True)

    #print(img)

    

    #coverting the type of pixel to float 32

    img = img.astype('float32')

    

    # appending the image into the list

    train_img.append(img)

    #print(train_img)

    

# converting the list to numpy array

train_x = np.array(train_img)

train_x.shape  
# defining the target

train_y = train['emergency_or_not'].values



# create validation set

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)

(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
# converting training images into torch format

train_x = train_x.reshape(1481, 3, 224, 224)

train_x  = torch.from_numpy(train_x)

print(type(train_x), train_x.size())


# converting the target into torch format

train_y = train_y.astype(int)

train_y = torch.from_numpy(train_y)



# converting validation images into torch format

val_x = val_x.reshape(165, 3, 224, 224)

val_x  = torch.from_numpy(val_x)



# converting the target into torch format

val_y = val_y.astype(int)

val_y = torch.from_numpy(val_y)
torch.manual_seed(0)



class Net(Module):   

    def __init__(self):

        super(Net, self).__init__()



        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer

            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2)

        )



        self.linear_layers = Sequential(

            Linear(32 * 56 * 56, 2)

        )



    # Defining the forward pass    

    def forward(self, x):

        x = self.cnn_layers(x)

        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)

        return x
# defining the model

model = Net()

# defining the optimizer

optimizer = Adam(model.parameters(), lr=0.0001)

# defining the loss function

criterion = CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()



print(model)
torch.manual_seed(0)



# batch size of the model

batch_size = 128



# number of epochs to train the model

n_epochs = 25



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

        

    permutation = torch.randperm(train_x.size()[0])



    training_loss = []

    for i in tqdm(range(0,train_x.size()[0], batch_size)):



        indices = permutation[i:i+batch_size]

        batch_x, batch_y = train_x[indices], train_y[indices]

        

        if torch.cuda.is_available():

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        

        optimizer.zero_grad()

        # in case you wanted a semi-full example

        outputs = model(batch_x)

        loss = criterion(outputs,batch_y)



        training_loss.append(loss.item())

        loss.backward()

        optimizer.step()

        

    training_loss = np.average(training_loss)

    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
# prediction for training set

prediction = []

target = []

permutation = torch.randperm(train_x.size()[0])

for i in tqdm(range(0,train_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = train_x[indices], train_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction.append(predictions)

    print(">>>>>>>>>>", type(predictions), type(batch_y))

    batch_y = batch_y.cpu().numpy()

    print(">>>>", type(predictions), type(batch_y))

    target.append(batch_y)

    

# training accuracy

accuracy = []

for i in range(len(prediction)):

    accuracy.append(accuracy_score(target[i],prediction[i]))

    

print('training accuracy: \t', np.average(accuracy))
# prediction for validation set

prediction_val = []

target_val = []

permutation = torch.randperm(val_x.size()[0])

for i in tqdm(range(0,val_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = val_x[indices], val_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction_val.append(predictions)

    batch_y=batch_y.cpu().numpy()

    target_val.append(batch_y)

    

# validation accuracy

accuracy_val = []

for i in range(len(prediction_val)):

    accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))

    

print('validation accuracy: \t', np.average(accuracy_val))
torch.manual_seed(0)



class Net(Module):   

    def __init__(self):

        super(Net, self).__init__()



        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer

            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            # dropout layer

            Dropout(),

            # Defining another 2D convolution layer

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            # dropout layer

            Dropout(),

        )



        self.linear_layers = Sequential(

            Linear(32 * 56 * 56, 2)

        )



    # Defining the forward pass    

    def forward(self, x):

        x = self.cnn_layers(x)

        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)

        return x

# defining the model

model = Net()

# defining the optimizer

optimizer = Adam(model.parameters(), lr=0.0001)

# defining the loss function

criterion = CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()



print(model)
torch.manual_seed(0)



# batch size of the model

batch_size = 128



# number of epochs to train the model

n_epochs = 25



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

        

    permutation = torch.randperm(train_x.size()[0])



    training_loss = []

    for i in tqdm(range(0,train_x.size()[0], batch_size)):



        indices = permutation[i:i+batch_size]

        batch_x, batch_y = train_x[indices], train_y[indices]

        

        if torch.cuda.is_available():

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        

        optimizer.zero_grad()

        # in case you wanted a semi-full example

        outputs = model(batch_x)

        loss = criterion(outputs,batch_y)



        training_loss.append(loss.item())

        loss.backward()

        optimizer.step()

        

    training_loss = np.average(training_loss)

    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
# prediction for training set

prediction = []

target = []

permutation = torch.randperm(train_x.size()[0])

for i in tqdm(range(0,train_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = train_x[indices], train_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction.append(predictions)

    batch_y=batch_y.cpu().numpy()

    target.append(batch_y)

    

# training accuracy

accuracy = []

for i in range(len(prediction)):

    accuracy.append(accuracy_score(target[i],prediction[i]))

    

print('training accuracy: \t', np.average(accuracy))
# prediction for validation set

prediction_val = []

target_val = []

permutation = torch.randperm(val_x.size()[0])

for i in tqdm(range(0,val_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = val_x[indices], val_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction_val.append(predictions)

    batch_y = batch_y.cpu().numpy()

    target_val.append(batch_y)

    

# validation accuracy

accuracy_val = []

for i in range(len(prediction_val)):

    accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))

    

print('validation accuracy: \t', np.average(accuracy_val))
torch.manual_seed(0)



class Net(Module):   

    def __init__(self):

        super(Net, self).__init__()



        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer

            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            # batch normalization layer

            BatchNorm2d(16),

            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),    

            # batch normalization layer

            BatchNorm2d(32),

            MaxPool2d(kernel_size=2, stride=2),

        )



        self.linear_layers = Sequential(

            Linear(32 * 56 * 56, 2)

        )



    # Defining the forward pass    

    def forward(self, x):

        x = self.cnn_layers(x)

        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)

        return x
# defining the model

model = Net()

# defining the optimizer

optimizer = Adam(model.parameters(), lr=0.00005)

# defining the loss function

criterion = CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()



print(model)
torch.manual_seed(0)



# batch size of the model

batch_size = 128



# number of epochs to train the model

n_epochs = 5



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

        

    permutation = torch.randperm(train_x.size()[0])



    training_loss = []

    for i in tqdm(range(0,train_x.size()[0], batch_size)):



        indices = permutation[i:i+batch_size]

        batch_x, batch_y = train_x[indices], train_y[indices]

        

        if torch.cuda.is_available():

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        

        optimizer.zero_grad()

        # in case you wanted a semi-full example

        outputs = model(batch_x)

        loss = criterion(outputs,batch_y)



        training_loss.append(loss.item())

        loss.backward()

        optimizer.step()

        

    training_loss = np.average(training_loss)

    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
# prediction for training set

prediction = []

target = []

permutation = torch.randperm(train_x.size()[0])

for i in tqdm(range(0,train_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = train_x[indices], train_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction.append(predictions)

    batch_y = batch_y.cpu().numpy()

    target.append(batch_y)

    

# training accuracy

accuracy = []

for i in range(len(prediction)):

    accuracy.append(accuracy_score(target[i],prediction[i]))

    

print('training accuracy: \t', np.average(accuracy))
# prediction for validation set

prediction_val = []

target_val = []

permutation = torch.randperm(val_x.size()[0])

for i in tqdm(range(0,val_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = val_x[indices], val_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction_val.append(predictions)

    batch_y=batch_y.cpu().numpy()

    target_val.append(batch_y)

    

# validation accuracy

accuracy_val = []

for i in range(len(prediction_val)):

    accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))

    

print('validation accuracy: \t', np.average(accuracy_val))
torch.manual_seed(0)



class Net(Module):   

    def __init__(self):

        super(Net, self).__init__()



        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer

            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            # adding batch normalization

            BatchNorm2d(16),

            MaxPool2d(kernel_size=2, stride=2),

            # adding dropout

            Dropout(),

            # Defining another 2D convolution layer

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),

            # adding batch normalization

            BatchNorm2d(32),

            MaxPool2d(kernel_size=2, stride=2),

            # adding dropout

            Dropout(),

        )



        self.linear_layers = Sequential(

            Linear(32 * 56 * 56, 2)

        )



    # Defining the forward pass    

    def forward(self, x):

        x = self.cnn_layers(x)

        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)

        return x
# defining the model

model = Net()

# defining the optimizer

optimizer = Adam(model.parameters(), lr=0.00025)

# defining the loss function

criterion = CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()



print(model)
torch.manual_seed(0)



# batch size of the model

batch_size = 128



# number of epochs to train the model

n_epochs = 10



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

        

    permutation = torch.randperm(train_x.size()[0])



    training_loss = []

    for i in tqdm(range(0,train_x.size()[0], batch_size)):



        indices = permutation[i:i+batch_size]

        batch_x, batch_y = train_x[indices], train_y[indices]

        

        if torch.cuda.is_available():

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        

        optimizer.zero_grad()

        # in case you wanted a semi-full example

        outputs = model(batch_x)

        loss = criterion(outputs,batch_y)



        training_loss.append(loss.item())

        loss.backward()

        optimizer.step()

        

    training_loss = np.average(training_loss)

    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
# prediction for training set

prediction = []

target = []

permutation = torch.randperm(train_x.size()[0])

for i in tqdm(range(0,train_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = train_x[indices], train_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction.append(predictions)

    batch_y=batch_y.cpu().numpy()

    target.append(batch_y)

    

# training accuracy

accuracy = []

for i in range(len(prediction)):

    accuracy.append(accuracy_score(target[i],prediction[i]))

    

print('training accuracy: \t', np.average(accuracy))
# prediction for validation set

prediction_val = []

target_val = []

permutation = torch.randperm(val_x.size()[0])

for i in tqdm(range(0,val_x.size()[0], batch_size)):

    indices = permutation[i:i+batch_size]

    batch_x, batch_y = val_x[indices], val_y[indices]



    if torch.cuda.is_available():

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()



    with torch.no_grad():

        output = model(batch_x.cuda())



    softmax = torch.exp(output).cpu()

    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)

    prediction_val.append(predictions)

    batch_y=batch_y.cpu().numpy()

    target_val.append(batch_y)

    

# validation accuracy

accuracy_val = []

for i in range(len(prediction_val)):

    accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))

    

print('validation accuracy: \t', np.average(accuracy_val))