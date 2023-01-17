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
#For reading and displaying images

from skimage.io import imread

import matplotlib.pyplot as plt

%matplotlib inline
#for creating validation set

from sklearn.model_selection import train_test_split

#for evaluating the model

from sklearn.metrics import accuracy_score

from tqdm import tqdm
#PyTorch libraries and modules

import torch

from torch.autograd import Variable

from torch.nn import Sequential, Conv2d, MaxPool2d, Module, Softmax, Linear, ReLU, BatchNorm2d, Dropout, CrossEntropyLoss

from torch.optim import SGD, Adam
#Loading the dataset

train = pd.read_csv('/kaggle/input/train_LbELtWX/train.csv')

test=pd.read_csv('/kaggle/input/test_ScVgIM0/test.csv')
#Loading the training images

train_img=[]

for img_name in tqdm(train['id']):

    #defining the img path

    image_path = '/kaggle/input/train_LbELtWX/train/' + str(img_name) + '.png'

    #reading the image

    img=imread(image_path, as_gray=True)

    #normalising the pixels value

    img/=img/255

    #converting the type of pixels to float

    img=img.astype('float32')

    #stacking the image to training set

    train_img.append(img)
#Converting the list to numpy array

train_x=np.array(train_img)



#Defining the target variable

train_y=train['label'].values

train_x.shape
#Visualising the training data

i=0

plt.figure(figsize=(8,8))

plt.subplot(221), plt.imshow(train_x[i+24], cmap='gray')

plt.subplot(222), plt.imshow(train_x[i+48], cmap='gray')

plt.subplot(223), plt.imshow(train_x[i+96], cmap='gray')

plt.subplot(224), plt.imshow(train_x[i+144], cmap='gray')


# create validation set

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)

(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
# converting training images into torch format

train_x = train_x.reshape(54000, 1, 28, 28)

train_x  = torch.from_numpy(train_x)



# converting the target into torch format

train_y = train_y.astype(int);

train_y = torch.from_numpy(train_y)



# shape of training data

train_x.shape, train_y.shape
# converting validation images into torch format

val_x = val_x.reshape(6000, 1, 28, 28)

val_x  = torch.from_numpy(val_x)



# converting the target into torch format

val_y = val_y.astype(int);

val_y = torch.from_numpy(val_y)



# shape of validation data

val_x.shape, val_y.shape


class Net(Module):   

    def __init__(self):

        super(Net, self).__init__()



        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer

            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),

            BatchNorm2d(4),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),

            BatchNorm2d(4),

            ReLU(inplace=True),

            MaxPool2d(kernel_size=2, stride=2),

        )



        self.linear_layers = Sequential(

            Linear(4 * 7 * 7, 10)

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

optimizer = Adam(model.parameters(), lr=0.07)

# defining the loss function

criterion = CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()

    

print(model)
def train(epoch):

    model.train()

    tr_loss = 0

    # getting the training set

    x_train, y_train = Variable(train_x), Variable(train_y)

    # getting the validation set

    x_val, y_val = Variable(val_x), Variable(val_y)

    # converting the data into GPU format

    if torch.cuda.is_available():

        x_train = x_train.cuda()

        y_train = y_train.cuda()

        x_val = x_val.cuda()

        y_val = y_val.cuda()



    # clearing the Gradients of the model parameters

    optimizer.zero_grad()

    

    # prediction for training and validation set

    output_train = model(x_train)

    output_val = model(x_val)



    # computing the training and validation loss

    loss_train = criterion(output_train, y_train)

    loss_val = criterion(output_val, y_val)

    train_losses.append(loss_train)

    val_losses.append(loss_val)



    # computing the updated weights of all the model parameters

    loss_train.backward()

    optimizer.step()

    tr_loss = loss_train.item()

    if epoch%2 == 0:

        # printing the validation loss

        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
# defining the number of epochs

n_epochs = 25

# empty list to store training losses

train_losses = []

# empty list to store validation losses

val_losses = []

# training the model

for epoch in range(n_epochs):

    train(epoch)
# plotting the training and validation loss

plt.plot(train_losses, label='Training loss')

plt.plot(val_losses, label='Validation loss')

plt.legend()

plt.show()
#prediction with the training set

with torch.no_grad():

    output=model(train_x.cuda())



softmax=torch.exp(output).cpu()

prob=list(softmax.numpy())

predictions=np.argmax(prob, axis=1)



#accuracy score on training set

accuracy_score(train_y, predictions)
# prediction for validation set

with torch.no_grad():

    output = model(val_x.cuda())



softmax = torch.exp(output).cpu()

prob = list(softmax.numpy())

predictions = np.argmax(prob, axis=1)



# accuracy on validation set

accuracy_score(val_y, predictions)