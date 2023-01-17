# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
import seaborn as sns
sns.countplot(y)
# Let's see some sample images
fig = plt.figure(figsize=(25,4))
fig.subplots_adjust(hspace=0.5)
for i,index in enumerate(np.random.randint(0,100,10)):
    ax = fig.add_subplot(2,5,i+1)
    ax.imshow(X[index].reshape(28,28), cmap='gray')
    ax.set_title("Label= {}".format(y[index]), fontsize = 20)
    ax.axis('off')
plt.show()
# Check IF some Feature variables are NaN
np.unique(np.isnan(X))[0]
# Check IF some Target Variables are NaN
np.unique(np.isnan(y))[0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0
import torch
import torch.utils.data
from torch.autograd import Variable
'''Create tensors for our train and test set. 
As you remember we need variable to accumulate gradients. 
Therefore first we create tensor, then we will create variable '''
# Numpy to Tensor Conversion (Train Set)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Numpy to Tensor Conversion (Train Set)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
# Make torch datasets from train and test sets
train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

# Create train and test data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)
import torch.nn as nn
import torch.nn.functional as F
class ANN(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(ANN, self).__init__()
    
        # Input Layer (784) -> 784
        self.fc1 = nn.Linear(input_dim, 784)
        # 784 -> 128
        self.fc2 = nn.Linear(784, 128)
        # 128 -> 128
        self.fc3 = nn.Linear(128, 128)
        # 128 -> 64
        self.fc4 = nn.Linear(128, 64)
        # 64 -> 64
        self.fc5 = nn.Linear(64, 64)
        # 64 -> 32
        self.fc6 = nn.Linear(64, 32)
        # 32 -> 32
        self.fc7 = nn.Linear(32, 32)
        # 32 -> output layer(10)
        self.output_layer = nn.Linear(32,10)
        # Dropout Layer (20%) to reduce overfitting
        self.dropout = nn.Dropout(0.2)
    
    # Feed Forward Function
    def forward(self, x):
        
        # flatten image input
        x = x.view(-1, 28 * 28)
        
        # Add ReLU activation function to each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        # Don't add any ReLU activation function to Last Output Layer
        x = self.output_layer(x)
        
        # Return the created model
        return x
# Create the Neural Network Model
model = ANN(input_dim = 784, output_dim = 10)
# Print its architecture
print(model)
import torch.optim as optim
# specify loss function
loss_fn = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9,nesterov = True)
# Define epochs (between 20-50)
epochs = 30

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

# Some lists to keep track of loss and accuracy during each epoch
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []



# Start epochs
for epoch in range(epochs):
    # monitor training loss
    train_loss = 0.0
    val_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # Set the training mode ON -> Activate Dropout Layers
    model.train() # prepare model for training
    # Calculate Accuracy         
    correct = 0
    total = 0
    
    # Load Train Images with Labels(Targets)
    for data, target in train_loader:
        
        # Convert our images and labels to Variables to accumulate Gradients
        data = Variable(data).float()
        target = Variable(target).type(torch.LongTensor)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # Calculate Training Accuracy 
        predicted = torch.max(output.data, 1)[1]        
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
    
        
    # Implement Validation like K-fold Cross-validation 
    # Set Evaluation Mode ON -> Turn Off Dropout
    model.eval() # Required for Evaluation/Test

    # Calculate Test/Validation Accuracy         
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:

            # Convert our images and labels to Variables to accumulate Gradients
            data = Variable(data).float()
            target = Variable(target).type(torch.LongTensor)

            # Predict Output
            output = model(data)

            # Calculate Loss
            loss = loss_fn(output, target)
            val_loss += loss.item()*data.size(0)
            # Get predictions from the maximum value
            predicted = torch.max(output.data, 1)[1]

            # Total number of labels
            total += len(target)

            # Total correct predictions
            correct += (predicted == target).sum()
    
    # calculate average training loss and accuracy over an epoch
    val_loss = val_loss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    
    # Put them in their list
    val_acc_list.append(accuracy)
    val_loss_list.append(val_loss)
    
    # Print the Epoch and Training Loss Details with Validation Accuracy   
    print('Epoch: {} \tTraining Loss: {:.4f}\t Val. acc: {:.2f}%'.format(
        epoch+1, 
        train_loss,
        accuracy
        ))
    # save model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        val_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = val_loss
    # Move to next epoch
    epoch_list.append(epoch + 1)
model.load_state_dict(torch.load('model.pt'))
plt.plot(epoch_list,train_loss_list)
plt.plot(val_loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Number of Epochs")
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
plt.plot(epoch_list,train_acc_list)
plt.plot(val_acc_list)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Accuracy vs Number of Epochs")
plt.legend(['Train', 'Test'], loc='best')
plt.show()
val_acc = sum(val_acc_list[20:]).item()/10
print("Test Accuracy of model = {} %".format(val_acc))
kaggle_test_set = pd.read_csv('../input/test.csv')

# Convert it to numpy array and Normalize it
kaggle_test_set = kaggle_test_set.values/255.0
kaggle_test_set = Variable(torch.from_numpy(kaggle_test_set)).float()
# Predicted Labels will be stored here
results = []

# Set Evaluation Mode ON -> Turn Off Dropout
model.eval() # Required for Evaluation/Test

with torch.no_grad():
    for image in kaggle_test_set:
        output = model(image)
        pred = torch.max(output.data, 1)[1]
        results.append(pred[0].numpy())
# Convert List to Numpy Array
results = np.array(results)
# Plot using Matplotlib
fig = plt.figure(figsize=(25,4))
fig.subplots_adjust(hspace=0.5)
for i,index in enumerate(np.random.randint(0,100,10)):
    ax = fig.add_subplot(2,5,i+1)
    ax.imshow(kaggle_test_set[index].reshape(28,28), cmap='gray')
    ax.set_title("Label= {}".format(results[index]), fontsize = 20)
    ax.axis('off')
plt.show()
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)
submission.head()