# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Churn_Modelling.csv')
kaggle_test_set = pd.read_csv('../input/test.csv')
dataset.head(5)
X = dataset.iloc[:,2:12].values
y = dataset.iloc[:,12].values
kaggle_test_set = kaggle_test_set.iloc[:,2:12].values
kaggle_test_set
y
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

kaggle_test_set[:, 1] = labelencoder_X_1.fit_transform(kaggle_test_set[:, 1])
kaggle_test_set[:, 2] = labelencoder_X_2.fit_transform(kaggle_test_set[:, 2])
kaggle_test_set = onehotencoder.fit_transform(kaggle_test_set).toarray()
kaggle_test_set.shape
# Avoid Dummy Variable Trap
X = X[:, 1:]
kaggle_test_set = kaggle_test_set[:, 1:]
X.shape, kaggle_test_set.shape
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
kaggle_test_set = sc.transform(kaggle_test_set)
X_train
X_test
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
class ANN(nn.Module):
    def __init__(self, input_dim = 11, output_dim = 1):
        super(ANN, self).__init__()
    
        # Input Layer (784) -> 784
        self.fc1 = nn.Linear(input_dim, 64)
        # 64 -> 64
        self.fc2 = nn.Linear(64, 64)
        # 64 -> 32
        self.fc3 = nn.Linear(64, 32)
        # 32 -> 32
        self.fc4 = nn.Linear(32, 32)
        # 32 -> output layer(10)
        self.output_layer = nn.Linear(32,1)
        # Dropout Layer (20%) to reduce overfitting
        self.dropout = nn.Dropout(0.2)
    
    # Feed Forward Function
    def forward(self, x):
        
        # Add ReLU activation function to each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # Don't add any ReLU activation function to Last Output Layer
        x = self.output_layer(x)
        
        # Return the created model
#         return F.softmax(x,dim=1)
        return nn.Sigmoid()(x)
# Create the Neural Network Model
model = ANN(input_dim = 11, output_dim = 1)
# Print its architecture
print(model)
import torch
import torch.utils.data
from torch.autograd import Variable
'''Create tensors for our train and test set. 
As you remember we need variable to accumulate gradients. 
Therefore first we create tensor, then we will create variable '''
# Numpy to Tensor Conversion (Train Set)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).view(-1,1)

# Numpy to Tensor Conversion (Train Set)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).view(-1,1)

kaggle_test_set = torch.from_numpy(kaggle_test_set)
y_train.shape
# Make torch datasets from train and test sets
train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

# Create train and test data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)
import torch.optim as optim
# specify loss function
loss_fn = nn.BCELoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9,nesterov = True)
# Define epochs (between 20-50)
epochs = 1000

# Some lists to keep track of loss and accuracy during each epoch
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# Set the training mode ON -> Activate Dropout Layers
model.train() # prepare model for training

for epoch in range(epochs):
    # monitor training loss
    train_loss = 0.0
    val_loss = 0.0
    
    ###################
    # train the model #
    ###################
    
    # Calculate Accuracy         
    correct = 0
    total = 0
    for data,target in train_loader:
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        #print("Target = ",target[0].item())
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        predicted = (torch.round(output.data[0]))
        # Total number of labels
        total += len(target)
        # Total correct predictions
        correct += (predicted == target).sum()

        # calculate the loss
        loss = loss_fn(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)

    # calculate average training loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    
    # Avg Accuracy
    accuracy = 100 * correct / float(total)
    # Put them in their list
    train_acc_list.append(accuracy)
    train_loss_list.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.4f}\t Acc: {:.2f}%'.format(
        epoch+1, 
        train_loss,
        accuracy
        ))
    # Move to next epoch
    epoch_list.append(epoch + 1)
import matplotlib.pyplot as plt
plt.plot(epoch_list,train_loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Number of Epochs")
plt.show()
plt.plot(epoch_list,train_acc_list)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Accuracy vs Number of Epochs")
plt.show()
correct = 0
total = 0
val_loss = 0
model.eval() # Required for Evaluation/Test
with torch.no_grad():
    for data, target in test_loader:

        # Convert our images and labels to Variables to accumulate Gradients
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)

        # Predict Output
        output = model(data)

        # Calculate Loss
        loss = loss_fn(output, target)
        val_loss += loss.item()*data.size(0)
        # Get predictions from the maximum value
        predicted = (torch.round(output.data[0]))

        # Total number of labels
        total += len(target)

        # Total correct predictions
        correct += (predicted == target).sum()
    
    # calculate average training loss and accuracy over an epoch
    val_loss = val_loss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
print("Accuracy = ",accuracy.item() * 0.01)

kaggle_test_set = Variable(kaggle_test_set).float()
# Predicted Labels will be stored here
results = []

# Set Evaluation Mode ON -> Turn Off Dropout
model.eval() # Required for Evaluation/Test

with torch.no_grad():
    for data in kaggle_test_set:
        output = model(data)
        pred = int((torch.round(output.data[0])).item())
        results.append(pred)
# Convert List to Numpy Array
results = np.array(results)
results
results = pd.Series(results,name="Exited")
submission = pd.concat([pd.Series(range(1,101),name = "ID"),results],axis = 1)
submission.to_csv("submission.csv",index=False)
submission.head(5)