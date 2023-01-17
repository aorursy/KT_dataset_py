# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print ("1. Loading data")
train = pd.read_csv("../input/train.csv",dtype = np.float32)
test  = pd.read_csv("../input/test.csv", dtype = np.float32)
print('split data into features(pixels) and labels(numbers from 0 to 9)')
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.10,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long
print('visualize one of the images in data set')
i = 100
plt.imshow(features_numpy[i].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[i]))
plt.savefig('graph.png')
plt.show()
print('2. Pytorch train and test sets')
batch_size = 100
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
splittest = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(splittest, batch_size = batch_size, shuffle = False)
    
print('3. Defining DL Model')
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
print('Implement a function for the validation pass')
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
print('Hyperparameters for our network')
input_size = 784
hidden_sizes = [392, 196, 128, 64, 31, 16]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                      ('relu3', nn.ReLU()),
                      ('fc4', nn.Linear(hidden_sizes[2], hidden_sizes[3])),
                      ('relu4', nn.ReLU()),
                      ('fc5', nn.Linear(hidden_sizes[3], hidden_sizes[4])),
                      ('relu5', nn.ReLU()),
                      ('fc6', nn.Linear(hidden_sizes[4], hidden_sizes[5])),
                      ('relu6', nn.ReLU()),
                      ('logits', nn.Linear(hidden_sizes[5], output_size))]))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.05)

print('4. Build DL Model with 30 Epochs')
epochs = 30
print_every = 40
steps = 0
# store loss and iteration
loss_list = []
iteration_list = []
accuracy_list = []
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(train_loader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_loader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(steps)
            accuracy_list.append(accuracy/len(test_loader))
            running_loss = 0
            
            # Make sure training is back on
            model.train()
print('DL Model: loss & accuracy visualization')
# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()
print ('5. Prepare test dataset for Predicton')
testfeatures_numpy = test.loc[:,test.columns != "label"].values/255 # normalization
testfeatures = torch.from_numpy(testfeatures_numpy)
#test_loader = torch.utils.data.DataLoader(dataset=test,
#                                           batch_size=batch_size, shuffle=False)
print ('Prediction on test set')
test_pred = torch.LongTensor()
y_pred = model(testfeatures)
_, predicted = torch.max(y_pred.data, 1)
test_pred = torch.cat((test_pred, predicted), dim=0)
print ('6. Generating submission file')
submission = pd.DataFrame(np.c_[np.arange(1, len(test)+1)[:,None], test_pred.numpy()], 
                      columns=['ImageId', 'Label'])
submission.to_csv('submission.csv', index=False, header=True)
