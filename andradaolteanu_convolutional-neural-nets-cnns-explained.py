# Original Image is formed by 3 channels

# Original pixels are the sum of R + G + B

print('Original Image top left corner:', 1+0+0, '\n')



# Filters: of size 3 x 3 containing trainable weights that identify patterns

print('R Output top left corner:', 1*0 + 0*1 + 1*0 + 0*1 + 3*1 + 2*2 + 1*1 + 1*0 + 1*0)

print('G Output top left corner:', 0*1 + 3*0 + 3*2 + 0*1 + 2*1 + 4*0 + 1*0 + 2*2 + 0*0)

print('B Output top left corner:', 0*1 + 0*1 + 3*1 + 0*0 + 1*0 + 1*2 + 2*1 + 3*1 + 1*2)
# Imports

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import transforms

from torchvision.datasets import MNIST

import torchvision.models # for alexnet model



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn.metrics

import seaborn as sns

import random

import os
def set_seed(seed = 1234):

    '''Sets the seed of the entire notebook so results are the same every time we run.

    This is for REPRODUCIBILITY.'''

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set

    torch.backends.cudnn.deterministic = True

    # Set a fixed value for the hash seed

    os.environ['PYTHONHASHSEED'] = str(seed)

    

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device available now:', device)
# Import an image example

image = plt.imread("../input/suki-image/suki.jpg")



# See Image and Shape

print('Image shape:', image.shape)

plt.imshow(image);
# Before aplying any convolutions we need to change the structure of the image:



# 1. Convert to Tensor

image = torch.from_numpy(image)

print('Image Tensor:', image.shape)



# 2. Bring the channel in front

image = image.permute(2, 0, 1)

print('Permuted Channel:', image.shape)



# Add 1 more dimension for batching (1 because we have only 1 image)

image = image.reshape([1, 3, 320, 320]).float()

print('Final image shape:', image.shape)
# Create 1 convolutional layer

# Padding and Stride are at default values

convolution_1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, padding=0, stride=1)



# Apply convolution to the image

conv1_image = convolution_1(image)

print('Convoluted Image shape:', conv1_image.shape)
# Select Convolution Parameters

conv_params = list(convolution_1.parameters())



print("len(conv_params):", len(conv_params))   # 2 sets of parameters in total

print("Filters:", conv_params[0].shape)        # there are 5 filters for 3 channels with size 5 by 5

print("Biases:", conv_params[1].shape)         # 5 biases for each filter
# Convert result to numpy

conv1_numpy_image = conv1_image.detach().numpy()



# Remove the dim 1 batch

conv1_numpy_image = conv1_numpy_image.reshape([5, 316, 316])



# Normalize to [0, 1] for plotting

maxim = np.max(conv1_numpy_image)

minim = np.min(conv1_numpy_image)



conv1_numpy_image = conv1_numpy_image - minim / (maxim - minim)



print('Image after Conv shape:', conv1_numpy_image.shape)



# Plotting the channels

plt.figure(figsize = (16, 5))



for i in range(5):

    plt.subplot(1, 5, i+1)

    plt.imshow(conv1_numpy_image[i]);
# We'll use the same image of Suki to continue with this example

# Create another convolutional layer (a second one)

convolution_2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=10, padding=2, stride=2)



# Apply convolution to the LAST convolution created

conv2_image = convolution_2(conv1_image)

print('Convoluted Image shape:', conv2_image.shape)
# Convert result to numpy

conv2_numpy_image = conv2_image.detach().numpy()



# Remove the dim 1 batch

conv2_numpy_image = conv2_numpy_image.reshape([8, 156, 156])



# Normalize to [0, 1] for plotting

maxim = np.max(conv2_numpy_image)

minim = np.min(conv2_numpy_image)



conv2_numpy_image = conv2_numpy_image - minim / (maxim - minim)



print('Image after Conv shape:', conv2_numpy_image.shape)



# Plotting the channels

plt.figure(figsize = (16, 5))



for i in range(8):

    plt.subplot(1, 8, i+1)

    plt.imshow(conv2_numpy_image[i]);
# Import alexNet

alexNet = torchvision.models.alexnet(pretrained=True)
# AlexNet Structure:

alexNet
# Creating the Architecture

class CNN_MNISTClassifier(nn.Module):                         # the class inherits from nn.Module

    def __init__(self):                                       # here we define the structure of the network

        super(CNN_MNISTClassifier, self).__init__()

        

        # Convolutional Layers that learn patterns in the data

        self.features = nn.Sequential(nn.Conv2d(1, 16, 3),      # output size: (28-3+0)/1 + 1 = 26

                                      nn.ReLU(),                # activation function

                                      nn.MaxPool2d(2, 2),       # 26/2 = 13

                                      nn.Conv2d(16, 10, 3),     # output size: (13-3+0)/1 + 1 = 11

                                      nn.ReLU(),                # activation function

                                      nn.MaxPool2d(2))          # 11/2 = 5

        

        # FNN which classifies the data

        self.classifier = nn.Sequential(nn.Linear(10*5*5, 128), # 10 channels * 5 by 5 pixel output

                                        nn.ReLU(),

                                        nn.Linear(128, 84), 

                                        nn.ReLU(),

                                        nn.Linear(84, 10))      # 10 possible predictions

        

    def forward(self, image, prints=False):                     # here we take the images through the network

        if prints: print('Original Image shape:', image.shape)

        

        # Take the image through the convolutions

        image = self.features(image)

        if prints: print('Convol Image shape:', image.shape)

        

        # Reshape the output to vectorize it

        image = image.view(-1, 10*5*5)

        if prints: print('Vectorized Image shape:', image.shape)

        

        # Log Probabilities output

        out = self.classifier(image)

        if prints: print('Out:', out)

            

        # Apply softmax

        out = F.log_softmax(out, dim=1)

        if prints: print('log_softmax(out):', out)

        

        return out
# Create Model Instance

model_example = CNN_MNISTClassifier()

model_example
# Importing the MNIST data

mnist_example = MNIST('data', train=True, download=True, transform=transforms.ToTensor())



# Select only 1 image (index=13)

image_example, label_example = list(mnist_example)[13]



# Add 1 more dimension for batching (1 image in the batch)

image_example = image_example.reshape([1, 1, 28, 28])
# Print Information

print('Image Example shape:', image_example.shape, '\n' +

      'Label Example:', label_example, '\n')



# Create Log Probabilities

out = model_example(image_example, prints=True)
# Create criterion and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_example.parameters(), lr = 0.001, weight_decay=0.0005)



# Compute loss

# Label Example has been transformed to tensor and reshaped so it suits the requirements of function

loss = criterion(out, torch.tensor(label_example).reshape(1))

print('Loss:', loss)



# Backpropagation

# Clears all gradients

optimizer.zero_grad()

# Compute gradients with respect to the loss

loss.backward()

# Update parameters

optimizer.step()
# Import the train and test data

# Transforms data to Tensors using `transforms`

mnist_train = MNIST('data', train = True, download=True, transform=transforms.ToTensor())

mnist_test = MNIST('data', train = False, download=True, transform=transforms.ToTensor())



# Select only first 500 instances from each to make training fast

mnist_train = list(mnist_train)[:500]

mnist_test = list(mnist_test)[:500]
def get_accuracy(model, data, batchSize = 20):

    '''Iterates through data and returnes average accuracy per batch.'''

    # Sets the model in evaluation mode

    model.eval()

    

    # Creates the dataloader

    data_loader = torch.utils.data.DataLoader(data, batch_size=batchSize)

    

    correct_cases = 0

    total_cases = 0

    

    for (images, labels) in iter(data_loader):

        # Is formed by 20 images (by default) with 10 probabilities each

        out = model(images)

        # Choose maximum probability and then select only the label (not the prob number)

        prediction = out.max(dim = 1)[1]

        # First check how many are correct in the batch, then we sum then convert to integer (not tensor)

        correct_cases += (prediction == labels).sum().item()

        # Total cases

        total_cases += images.shape[0]

    

    return correct_cases / total_cases
def train_network(model, train_data, test_data, batchSize=20, num_epochs=1, learning_rate=0.01, weight_decay=0,

                 show_plot = True, show_acc = True):

    

    '''Trains the model and computes the average accuracy for train and test data.

    If enabled, it also shows the loss and accuracy over the iterations.'''

    

    print('Get data ready...')

    # Create dataloader for training dataset - so we can train on multiple batches

    # Shuffle after every epoch 

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)

    

    # Create criterion and optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    

    # Losses & Iterations: to keep all losses during training (for plotting)

    losses = []

    iterations = []

    # Train and test accuracies: to keep their values also (for plotting)

    train_acc = []

    test_acc = []

    

    print('Training started...')

    iteration = 0

    # Train the data multiple times

    for epoch in range(num_epochs):

        

        for images, labels in iter(train_loader):

            # Set model in training mode:

            model.train()

            

            # Create log probabilities

            out = model(images)

            # Clears the gradients from previous iteration

            optimizer.zero_grad()

            # Computes loss: how far is the prediction from the actual?

            loss = criterion(out, labels)

            # Computes gradients for neurons

            loss.backward()

            # Updates the weights

            optimizer.step()

            

            # Save information after this iteration

            iterations.append(iteration)

            iteration += 1

            losses.append(loss)

            # Compute accuracy after this epoch and save

            train_acc.append(get_accuracy(model, train_data))

            test_acc.append(get_accuracy(model, test_data))

            

    

    # Show Accuracies

    # Show the last accuracy registered

    if show_acc:

        print("Final Training Accuracy: {}".format(train_acc[-1]))

        print("Final Testing Accuracy: {}".format(test_acc[-1]))

    

    # Create plots

    if show_plot:

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)

        plt.title("Loss Curve")

        plt.plot(iterations[::20], losses[::20], label="Train", linewidth=4, color='#7100FF')

        plt.xlabel("Iterations")

        plt.ylabel("Loss")



        plt.subplot(1,2,2)

        plt.title("Accuracy Curve")

        plt.plot(iterations[::20], train_acc[::20], label="Train", linewidth=4, color='#FFD800')

        plt.plot(iterations[::20], test_acc[::20], label="Test", linewidth=4, color='#FF8B00')

        plt.xlabel("Iterations")

        plt.ylabel("Accuracy")

        plt.legend(loc='best')

        plt.show()
# Create Model Instance

model1 = CNN_MNISTClassifier()



# Train...

train_network(model1, mnist_train, mnist_test, num_epochs=200)
def get_confusion_matrix(model, test_data):

    # First we make sure we disable Gradient Computing

    torch.no_grad()

    

    # Model in Evaluation Mode

    model.eval()

    

    preds, actuals = [], []



    for image, label in mnist_test:

        # Append 1 more dimension for batching

        image = image.reshape(1, 1, 28, 28)

        # Prediction

        out = model(image)



        prediction = torch.max(out, dim=1)[1].item()

        preds.append(prediction)

        actuals.append(label)

    

    return sklearn.metrics.confusion_matrix(preds, actuals)
plt.figure(figsize=(16, 5))

sns.heatmap(get_confusion_matrix(model1, mnist_test), cmap='icefire', annot=True, linewidths=0.1,

           fmt=',')

plt.title('Confusion Matrix: CNN', fontsize=15);