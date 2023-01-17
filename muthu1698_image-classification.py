# importing the libraries



import numpy as np

import pandas as pd

import torch

import torchvision

import matplotlib.pyplot as plt

from time import time

from torchvision import datasets, transforms

from torch import nn, optim
# transformations to be applied on images



transform = transforms.Compose([transforms.ToTensor(),

                              transforms.Normalize((0.5,), (0.5,)),

                              ])
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)

testset = datasets.MNIST('./', download=True, train=False, transform=transform)
# defining trainloader and testloader

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# shape of training data

dataiter = iter(trainloader)

images, labels = dataiter.next()



print(images.shape)

print(labels.shape)
# visualizing the training images

plt.imshow(images[0].numpy().squeeze(), cmap='gray')
# shape of validation data

dataiter = iter(testloader)

images, labels = dataiter.next()



print(images.shape)

print(labels.shape)
# defining the model architecture

class Net(nn.Module):   

  def __init__(self):

      super(Net, self).__init__()



      self.cnn_layers = nn.Sequential(

          # Defining a 2D convolution layer

          nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),

          nn.BatchNorm2d(4),

          nn.ReLU(inplace=True),

          nn.MaxPool2d(kernel_size=2, stride=2),

          # Defining another 2D convolution layer

          nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),

          nn.BatchNorm2d(4),

          nn.ReLU(inplace=True),

          nn.MaxPool2d(kernel_size=2, stride=2),

      )



      self.linear_layers = nn.Sequential(

          nn.Linear(4 * 7 * 7, 10)

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

optimizer = optim.Adam(model.parameters(), lr=0.01)

# defining the loss function

criterion = nn.CrossEntropyLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()

    

print(model)
for i in range(10):

    running_loss = 0

    for images, labels in trainloader:



        if torch.cuda.is_available():

          images = images.cuda()

          labels = labels.cuda()



        # Training pass

        optimizer.zero_grad()

        

        output = model(images)

        loss = criterion(output, labels)

        

        #This is where the model learns by backpropagating

        loss.backward()

        

        #And optimizes its weights here

        optimizer.step()

        

        running_loss += loss.item()

    else:

        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))


# getting predictions on test set and measuring the performance

correct_count, all_count = 0, 0

for images,labels in testloader:

  for i in range(len(labels)):

    if torch.cuda.is_available():

        images = images.cuda()

        labels = labels.cuda()

    img = images[i].view(1, 1, 28, 28)

    with torch.no_grad():

        logps = model(img)



    

    ps = torch.exp(logps)

    probab = list(ps.cpu()[0])

    pred_label = probab.index(max(probab))

    true_label = labels.cpu()[i]

    if(true_label == pred_label):

      correct_count += 1

    all_count += 1



print("Number Of Images Tested =", all_count)

print("\nModel Accuracy =", (correct_count/all_count))
# importing the libraries

import tensorflow as tf



from tensorflow.keras import datasets, layers, models

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path='mnist.npz')

# Normalize pixel values to be between 0 and 1

train_images, test_images = train_images / 255.0, test_images / 255.0
# visualizing a few images

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap='gray')

plt.show()
# shape of the training and test set

(train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape)
# reshaping the images

train_images = train_images.reshape((60000, 28, 28, 1))

test_images = test_images.reshape((10000, 28, 28, 1))



# one hot encoding the target variable

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
# defining the model architecture

model = models.Sequential()

model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Conv2D(4, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='softmax'))
# summary of the model

model.summary()
# compiling the model

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# training the model

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))