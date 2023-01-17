# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
x = torch.tensor([2,3,3.])
x
import torch.nn as nn
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
from torch.utils.data import DataLoader, TensorDataset
train_ds = TensorDataset(inputs, targets)
# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
def sigmoid(x):
    return 1/(1-torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)
bais = torch.randn((1,1))
y_prob = sigmoid(torch.sum(features*weights) + bais)
y_prob, features, weights, bais
y_prob = sigmoid(torch.mm(features,weights.t()) + bais)
y_prob
y_prob = sigmoid(torch.matmul(features,weights.t()) + bais)
y_prob, weights.t()
y_prob = sigmoid(torch.matmul(features,weights.reshape(5,1)) + bais)
y_prob, weights.reshape(5,1)
y_prob = sigmoid(torch.matmul(features,weights.resize(5,1)) + bais)
y_prob, weights.reshape(5,1)
y_prob = sigmoid(torch.matmul(features,weights.view(5,1)) + bais)
y_prob, weights.reshape(5,1)
featutes = torch.randn((1,3))
n_input = featutes.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

b1 = torch.randn(1, n_hidden)
b2 = torch.randn(1, n_output)

h = sigmoid(torch.mm(featutes, W1)+b1)
y = sigmoid(torch.mm(h, W2)+b2)
y, h, W1, W2, b1, b2
%matplotlib inline

import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, label = next(dataiter)
print(type(images))
print(images.shape)
print(label.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
import torch

def sigmoid(x):
    return 1/(1-torch.exp(-x))

n_input = 784
n_hidden = 256
n_output = 10

inputs = images.view(images.shape[0], -1)

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

b1 = torch.randn(n_hidden)
b2 = torch.randn(n_output)

h = sigmoid(torch.mm(inputs, W1)+b1)
y = torch.mm(h, W2)+b2

def softmax(x):
    return torch.exp(x)/(torch.sum(torch.exp(x), dim=1).view(-1,1))

prob = softmax(y)
print(prob.shape)
print(prob.sum(dim=1))
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

        self.sigmoid =nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.softmax(self.output(self.sigmoid(self.hidden(x))))
model = Network()
model.forward(inputs)
import torch.nn.functional as F

class Network_1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return F.softmax(self.output(torch.sigmoid(self.hidden(x))), dim=1)
model1 = Network_1()
model1.forward(inputs)
class Network_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x))
        return x
model2 = Network_2()
x = model2.forward(inputs)
#this shows how we can define a Sequential Model
model = nn.Sequential(nn.Linear(784, 128),
                     nn.ReLU(),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Linear(64, 10))

cretion = nn.CrossEntropyLoss()
loss = cretion(x, label)
print(loss)
#this shows how we can define a Sequential Model
model = nn.Sequential(nn.Linear(784, 128),
                     nn.ReLU(),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Linear(64, 10),
                     nn.LogSoftmax(dim=1))

cretion = nn.CrossEntropyLoss()
x=model(inputs)
loss = cretion(x, label)
print(loss)
torch.set_grad_enabled(True)
data = torch.randn((2,2), requires_grad=True)
y=data**2
y.grad_fn
z=y.mean()
z.grad_fn
z.backward()
print(data.grad)
print(data/2)
#this shows how we can define a Sequential Model
model = nn.Sequential(nn.Linear(784, 128),
                     nn.ReLU(),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Linear(64, 10),
                     nn.LogSoftmax(dim=1))

cretion = nn.CrossEntropyLoss()
x=model(inputs)
loss = cretion(x, label)
print(loss)

print("Before",model[0].weight.grad)
loss.backward()
print("After", model[0].weight.grad)
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)

print("Initial Weigts:---", model[0].weight)

optimizer.zero_grad()

output=model.forward(inputs)
loss = cretion(output, label)
loss.backward()
print("Gradinet--", model[0].weight.grad)
optimizer.step()
print("Updated weights --",model[0].weight)
model = nn.Sequential(nn.Linear(784, 128),
                     nn.ReLU(),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Linear(64, 10),
                     nn.LogSoftmax(dim=1))

cretion = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)
epoch = 5
for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        inputs = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output=model.forward(inputs)
        loss = cretion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        print(f"Training Loss:{running_loss/len(trainloader)}")
model = nn.Sequential(nn.Linear(784, 128),
                     nn.ReLU(),
                     nn.Linear(128, 64),
                     nn.ReLU(),
                     nn.Linear(64, 10),
                     nn.LogSoftmax(dim=1))

cretion = torch.nn.NLLLoss()

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)
epoch = 5
for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        inputs = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output=model.forward(inputs)
        loss = cretion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        print(f"Training Loss:{running_loss/len(trainloader)}")
%matplotlib inline

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logits = model.forward(img)
    
ps = F.softmax(logits, dim=1)
view_classify(img.view(1,28,28), ps)
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),])
trainset = datasets.FashionMNIST('FashionMNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print(trainset)
testset = datasets.FashionMNIST('FashionMNIST_data/test/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
testset, testloader
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 5

for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        images=images.view(images.shape[0], -1)
        inp = model(images)
        loss = criterion(inp, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    else:
        print(f"Training Loss:{running_loss}")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    

images, labels = next(iter(testloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logits = model.forward(img)
    
ps = torch.exp(logits)
view_classify(img.view(1,28,28), ps, version='Fashion')
top_p, top_class = ps.topk(1, dim=1)
print(f"Predicted Class is: {top_class[0][0]}")
print(f"Predicted Class Probability: {top_p[0][0]}")
equals = top_class == labels
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f"Accuracy: {accuracy.item()*100}%")
epoch = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        images=images.view(images.shape[0], -1)
        inp = model(images)
        loss = criterion(inp, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in trainloader:
                inp=images.view(images.shape[0], -1)
                log_ps = model.forward(inp)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class = labels
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        
        print("Epoch: {}/{}..".format(e+1, epoch),
             "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
              "Test Loss {:.3f}..".format(test_loss/len(testloader)),
              "Test Accuarcy: {:.3f}..".format(accuracy/len(testloader)))
            
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")
plt.legend(frameon=False)
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
model
epoch = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        images=images.view(images.shape[0], -1)
        inp = model(images)
        loss = criterion(inp, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in trainloader:
                inp=images.view(images.shape[0], -1)
                log_ps = model.forward(inp)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class = labels
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        
        print("Epoch: {}/{}..".format(e+1, epoch),
             "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
              "Test Loss {:.3f}..".format(test_loss/len(testloader)),
              "Test Accuarcy: {:.3f}..".format(accuracy/len(testloader)))
            
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")
plt.legend(frameon=False)
print("our Model: \n\n", model, "\n")
print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.path')
state_dict= torch.load('checkpoint.path')
print(state_dict.keys())
model.load_state_dict(state_dict)
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
test_dataset = datasets.ImageFolder("../input/cat-and-dog/test_set", transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
transform = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
train_dataset = datasets.ImageFolder("../input/cat-and-dog/training_set", transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

data_iter = iter(trainloader)

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    imshow(images[ii], ax=ax, normalize=False)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
data_iter = iter(testloader)

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    imshow(images[ii], ax=ax, normalize=False)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_dataset = datasets.ImageFolder("../input/cat-and-dog/test_set", transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

train_dataset = datasets.ImageFolder("../input/cat-and-dog/test_set", transform=train_transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
model = models.densenet121(pretrained=True)
model
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        #steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if True:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()