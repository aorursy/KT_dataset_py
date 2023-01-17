!pip3 install jupyterthemes
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from colorama import Fore, Style

import os

import sys

from jupyterthemes import jtplot

jtplot.style(theme="monokai", context="notebook", ticks=True)
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df.head(5)
print(Fore.YELLOW, "Loading data information ...", Style.RESET_ALL)

df.info()
# Check class distribution

sns.countplot(x="class", data=df)
for cols in df.columns:

    unique_values = df[cols].unique()

    print(Fore.YELLOW, f"Number of unique values in '{cols}':", Style.RESET_ALL, len(unique_values))
X, Y = df.drop("class", axis=1), df["class"]
import torch

import torch.nn as nn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder





label_encoder = LabelEncoder()

for i in X.columns:

    X[i] = label_encoder.fit_transform(X[i])

    

label_encoder = LabelEncoder()

Y = label_encoder.fit_transform(Y)

X.head()
Y
X = pd.get_dummies(X, columns=X.columns, drop_first=True)

X.head(5)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler



stsc = StandardScaler()

Xtrain = stsc.fit_transform(Xtrain)

Xtest = stsc.transform(Xtest)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

Xtrain = pca.fit_transform(Xtrain)

Xtest = pca.transform(Xtest)
D = X.shape[1]



Ytrain = Ytrain.reshape(-1, 1)

Ytest = Ytest.reshape(-1, 1)



print(Fore.YELLOW, "Shapes for Training Data....", Style.RESET_ALL)

print(f"Shape of Xtrain : {Xtrain.shape}")

print(f"Shape of Ytrain : {Ytrain.shape}")





print(Fore.BLUE, "Shapes for Testing Data....", Style.RESET_ALL)

print(f"Shape of Xtest : {Xtest.shape}")

print(f"Shape of Ytest : {Ytest.shape}")
print(Fore.YELLOW, "Creating PyTorch Datasets for computation")



train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain.astype(np.float32)), torch.from_numpy(Ytrain.astype(np.float32)))

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtest.astype(np.float32)), torch.from_numpy(Ytest.astype(np.float32)))
batch_size = 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class Logistic(nn.Module):

    

    def __init__(self, n_units, n_classes):

        

        super(Logistic, self).__init__()

        

        self.seq = nn.Sequential(nn.Linear(n_units, n_classes), nn.Sigmoid())

        

    def forward(self, X):

        X = self.seq(X)

        

        return X
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
logregmodel = Logistic(Xtrain.shape[1], 1)

logregmodel.to(device)
# Define the loss and optimizer

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(logregmodel.parameters())
# Define the training loop 

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=20):

    

    """

    ----------------------------------------------------

    Description : Function to do batch gradient descent 

                  on the input dataset

                  

    Arguments :

    

    model -- a pytorch model 

    criterion -- a pytorch module which contains the loss

    optimizer -- a pytorcch module which contains the optimizers used for batch gradient descent

    train_loader -- a pytorch dataloader representing the training set

    test_loader -- a pytorch dataloader representing the testing set

    epochs -- an integer representing the number of training loops to go through

    

    Return :

    

    train_losses -- a numpy array containing the loss values encountered during training

    test_losses -- a numpy array containing the loss values encountered during validation

    

    Usage :

    

    train_loss, test_val = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=10000)

    

    -------------------------------------------------------    

    

    """

    

    train_losses = np.zeros(epochs)

    test_losses = np.zeros(epochs)

    

    for epoch in range(epochs):

        

        train_loss = []

        

        for inputs, targets in train_loader:

            

            # Move the inputs and targets to the device

            inputs, targets = inputs.to(device), targets.to(device)

            

            # Zero-initiialize the optimizer gradients

            optimizer.zero_grad()

            

            # Forward pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            

            # Backward and optimize

            loss.backward()

            optimizer.step()

            

            

            train_loss.append(loss.item())

            

        

        test_loss = []

        

        for inputs, targets in test_loader:

            

            # Move the inputs and targets to the device

            inputs, targets = inputs.to(device), targets.to(device)

            

            # Forward pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            

            test_loss.append(loss.item())

            

        

        train_loss = np.mean(train_loss)

        test_loss = np.mean(test_loss)

        

        train_losses[epoch] = train_loss

        test_losses[epoch] = test_loss

        

        print(f"Epoch : {epoch+1}/{epochs} | Train Loss : {train_loss} | Test Loss : {test_loss}")

            

        

    return train_losses, test_losses

            

            
train_losses, test_losses = batch_gd(logregmodel, criterion, optimizer, train_loader, test_loader, epochs=200)
plt.title("Epochs vs Losses")

plt.plot(train_losses, label="Train losses")

plt.plot(test_losses, label="Test losses")

plt.xlabel("Epochs")

plt.ylabel("Losses")

plt.legend()

plt.show()
# Get the model accuracy



def get_accuracy(evalmodel, train_loader, test_loader):

    

    """

    -----------------------------------------------

    Description : To calculate the accuracy rate of the model

    

    Arguments :

    

    model : a pytorch model 

    train_loader : a pytorch data loader representing the training set

    test_loader : a pytorch data loader representing the testing set

    

    Return:

    

    train_acc : a float value representing the training accuracy of the model

    test_acc : a float value representing the testing accuracy of the model

    

    

    Usage :

    

    trainAcc, testAcc = get_accuracy(model, train_loader, test_loader)

    --------------------------------------------------

    

    """

    

    

    n_correct = 0

    n_total = 0

    

    for inputs, targets in train_loader:

        

        # move targets to the device

        inputs, targets = inputs.to(device), targets.to(device)

        

        # Forward pass

        outputs = evalmodel(inputs).detach().numpy()

        

        n_correct += np.mean(targets.detach().numpy() == np.round(outputs))

        

        n_total += 1

        

    

    train_acc = n_correct / n_total

    

    

    for inputs, targets in test_loader:

        

        # move targets to the device

        inputs, targets = inputs.to(device), targets.to(device)

        

        # Forward pass

        outputs = evalmodel(inputs).detach().numpy()

        

        n_correct += np.mean(targets.detach().numpy() == np.round(outputs))

        n_total += 1

        

    

    test_acc = n_correct / n_total

    

    

    return train_acc, test_acc

        

        
train_acc , test_acc = get_accuracy(logregmodel, train_loader, test_loader)



print(f"Training Accuracy : {train_acc} || Testing Accuracy : {test_acc}")
class ANN(nn.Module):

    

    def __init__(self, n_features, n_classes):

        

        super(ANN, self).__init__()

        

        self.dense = nn.Sequential(

                nn.Linear(n_features, 20),

                nn.ReLU(),

                nn.Linear(20, 10),

                nn.ReLU(),

                nn.Linear(10, n_classes),

                nn.Sigmoid()

        )

        

    def forward(self, X):

        

        X = self.dense(X)

        

        return X
annmodel = ANN(Xtrain.shape[1], 1)

annmodel.to(device)
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(annmodel.parameters())
train_losses, test_losses = batch_gd(annmodel, criterion, optimizer, train_loader, test_loader, epochs=100)
plt.title("Epochs vs Losses")

plt.plot(train_losses, label="Training loss")

plt.plot(test_losses, label="Test loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
train_acc, test_acc = get_accuracy(annmodel, train_loader, test_loader)



print(f"Training Acc : {train_acc} | Test Acc : {test_acc}")