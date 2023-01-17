# Import necessary packages



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import torch



import helper



import matplotlib.pyplot as plt



from torch import nn, optim

from torch.autograd import Variable
### Run this cell



from torchvision import datasets, transforms



# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                              transforms.Normalize((0.5,), (0.5,)),

                              ])

# Download and load the training data

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()

print(type(images))

print(images.shape)

print(labels.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
## Solution

def activation(x):

    return 1/(1+torch.exp(-x))



# Flatten the input images

inputs = images.view(images.shape[0], -1)



# Create parameters

w1 = torch.randn(784, 256)

b1 = torch.randn(256)



w2 = torch.randn(256, 10)

b2 = torch.randn(10)



h = activation(torch.mm(inputs, w1) + b1)



out = torch.mm(h, w2) + b2


def softmax(x):

    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)



probabilities = softmax(out)



# Does it have the right shape? Should be (64, 10)

print(probabilities.shape)

# Does it sum to 1?

print(probabilities.sum(dim=1))
from torch import nn
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        

        # Inputs to hidden layer linear transformation

        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 units - one for each digit

        self.output = nn.Linear(256, 10)

        

        # Define sigmoid activation and softmax output 

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):

        # Pass the input tensor through each of our operations

        x = self.hidden(x)

        x = self.sigmoid(x)

        x = self.output(x)

        x = self.softmax(x)

        

        return x
# Create the network and look at it's text representation

model = Network()

model
import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self):

        super().__init__()

        # Inputs to hidden layer linear transformation

        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 units - one for each digit

        self.output = nn.Linear(256, 10)

        

    def forward(self, x):

        # Hidden layer with sigmoid activation

        x = F.sigmoid(self.hidden(x))

        # Output layer with softmax activation

        x = F.softmax(self.output(x), dim=1)

        

        return x
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        # Defining the layers, 128, 64, 10 units each

        self.fc1 = nn.Linear(784, 128)

        self.fc2 = nn.Linear(128, 64)

        # Output layer, 10 units - one for each digit

        self.fc3 = nn.Linear(64, 10)

        

    def forward(self, x):

        ''' Forward pass through the network, returns the output logits '''

        

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)

        x = F.softmax(x, dim=1)

        

        return x



model = Network()

model
print(model.fc1.weight)

print(model.fc1.bias)
# Set biases to all zeros

model.fc1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01

model.fc1.weight.data.normal_(std=0.01)
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
# Grab some data 

dataiter = iter(trainloader)

images, labels = dataiter.next()



# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 

images.resize_(64, 1, 784)

# or images.resize_(images.shape[0], 1, 784) to automatically get batch size



# Forward pass through the network

img_idx = 0

ps = model.forward(images[img_idx,:])



img = images[img_idx]

view_classify(img.view(1, 28, 28), ps)
# Hyperparameters for our network

input_size = 784

hidden_sizes = [128, 64]

output_size = 10



# Build a feed-forward network

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),

                      nn.ReLU(),

                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),

                      nn.ReLU(),

                      nn.Linear(hidden_sizes[1], output_size),

                      nn.Softmax(dim=1))

print(model)



# Forward pass through the network and display output

images, labels = next(iter(trainloader))

images.resize_(images.shape[0], 1, 784)

ps = model.forward(images[0,:])

view_classify(images[0].view(1, 28, 28), ps)
print(model[0])

model[0].weight
from collections import OrderedDict

model = nn.Sequential(OrderedDict([

                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),

                      ('relu1', nn.ReLU()),

                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),

                      ('relu2', nn.ReLU()),

                      ('output', nn.Linear(hidden_sizes[1], output_size)),

                      ('softmax', nn.Softmax(dim=1))]))

model
print(model[0])

print(model.fc1)