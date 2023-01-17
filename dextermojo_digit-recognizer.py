!pip3 install jupyterthemes
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from colorama import Fore, Style

import torchvision

import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix

import itertools

import os

import sys

from jupyterthemes import jtplot

jtplot.style(theme="monokai", context="notebook", ticks=True)
train_dataset = torchvision.datasets.MNIST(root=".", train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root=".", train=False, transform=transforms.ToTensor(), download=True)
train_dataset.data
train_dataset.targets
num_classes = len(set(train_dataset.targets.numpy()))

print(Fore.YELLOW, "Number of classes : ", Style.RESET_ALL, num_classes)
batch_size = 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class CNN(nn.Module):

    

    def __init__(self, n_classes):

        

        super(CNN, self).__init__()

        

        self.conv_layers = nn.Sequential(

                        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),

                        nn.ReLU(),

                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),

                        nn.ReLU(),

                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),

                        nn.ReLU())

        

        self.dense = nn.Sequential(

                    nn.Dropout(0.2),

                    nn.Linear(128*2*2, 512),

                    nn.ReLU(),

                    nn.Dropout(0.2),

                    nn.Linear(512, n_classes))

        

        

    

    def forward(self, X):

        

        out = self.conv_layers(X)

        out = out.view(out.size(0), -1)

        out = self.dense(out)

        return out
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
model = CNN(n_classes=num_classes)

model = model.to(device)

print(model)
# Define the loss function and the optimizers

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=100):

    

    """

    

    -----------------------------------------------------------------

    Description : Computes the batch gradient descent on the data

    

    Arguments :

    

    model --- a pytorch initialized model

    criterion --- a pytorch initialized loss function

    optimizer --- a pytorch initialized optimizer function

    train_loader --- a pytorch initialized data loader representing training data

    test_loader --- a pytorch initialized data loader representing test data

    epochs --- total number of training loops to process

    

    

    Return :

    

    train_losses --- a numpy nd-array representing the training losses

    test_losses --- a numpy nd-array representing the validation losses

    

    

    Usage :

    

    TrainLoss, TestLoss = batch_gd(model, critierion, optimizer, train_loader, test_loader, epochs=10)

    

    In case you want to epoch for 100 loops then just 

    

    TrainLoss, TestLoss = batch_gd(model, critierion, optimizer, train_loader, test_loader)

    --------------------------------------------------------------------

    

    """

    

    train_losses = np.zeros(epochs)

    test_losses = np.zeros(epochs)

    

    for epoch in range(epochs):

        

        train_loss = []

        

        for inputs, targets in train_loader:

            

            # Move inputs and targets to device 

            inputs, targets = inputs.to(device), targets.to(device)

            

            

            # Zero the optimizer gradients

            optimizer.zero_grad()

            

            

            # Forward pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            

            # Backward and optimize

            loss.backward()

            optimizer.step()

            

            # Track the training loss

            train_loss.append(loss.item())

            

        

        test_loss = []

        

        for inputs, targets in test_loader:

            

            # Move inputs and targets to device

            inputs, targets = inputs.to(device), targets.to(device)

            

            # Forward pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            

            # Track the validation loss

            test_loss.append(loss.item())

            

            

        train_loss = np.mean(train_loss)

        test_loss = np.mean(test_loss)

        

        train_losses[epoch] = train_loss

        test_losses[epoch] = test_loss

        

        

        print(f"Epoch : {epoch+1}/{epochs} || Train Loss : {train_loss} || Test Loss : {test_loss}")

        

    

    return train_losses, test_losses

                 

            
train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader)
plt.title("Epochs vs Loss")

plt.plot(train_losses, label="Train Loss")

plt.plot(test_losses, label="Test Loss")

plt.xlabel("Epochs")

plt.ylabel("Losses")

plt.legend()

plt.show()
def get_model_accuracy(model, loader):

    

    n_correct = 0

    n_total = 0

    

    for inputs, targets in loader:

        

        inputs , targets = inputs.to(device), targets.to(device)

        

        outputs = model(inputs)

        

        _, predictions = torch.max(outputs , 1)

        

        n_correct += (predictions == targets).sum().item()

        n_total += targets.shape[0]

        

    return n_correct / n_total
train_acc = get_model_accuracy(model, train_loader)

test_acc = get_model_accuracy(model, test_loader)



print(f"Training Accuracy : {train_acc} || Validation Accuracy : {test_acc}")
test_df = pd.read_csv("../input/digit-recognizer/test.csv")

test_df.head(5)
X = test_df.values

X = X.reshape(-1, 1, 28, 28)

print(f"Shape of Test data : {X.shape}")
inputs = np.asarray(X)

inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)



outputs = model(inputs)

_, predictions = torch.max(outputs, 1)

submission_predictions = predictions.detach().cpu().numpy()



    

# print(submission_predictions)

submission_df = pd.DataFrame().from_dict({"ImageId":[i+1 for i in range(len(submission_predictions))], "Label":submission_predictions})

submission_df.head(5)

    

    
submission_df.to_csv("submission.csv", index=False)
submission_df