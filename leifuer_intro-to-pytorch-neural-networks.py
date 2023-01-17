# Import necessary packages



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import torch



import helper



import matplotlib.pyplot as plt
### Run this cell



from torchvision import datasets, transforms



# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                              transforms.Normalize((0.5,), (0.5,)),

                              ])



# Download and load the training data

trainset = datasets.MNIST('/MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()

print(type(images))

print(images.shape)

print(labels.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
## Your solution

def activation(x):

    """ Sigmoid activation function 

    

        Arguments

        ---------

        x: torch.Tensor

    """

    return 1/(1+torch.exp(-x))



inputs = images.reshape(64, 784)



W1 = torch.randn(784, 256)

B1 = torch.randn(256)



W2 = torch.randn(256,10)

B2 = torch.randn(10)



H = activation(torch.mm(inputs, W1) + B1)



out = torch.mm(H, W2) + B2

# output of your network, should have shape (64,10)

out.shape
def softmax(x):

    ## TODO: Implement the softmax function here

    return torch.exp(x) / (torch.sum(torch.exp(x), dim=1).view(-1, 1))



# Here, out should be the output of the network in the previous excercise with shape (64,10)

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
## Your solution here

class My_Network(nn.Module):

    def __init__(self):

        super().__init__()

        # Inputs to 1st hidden layer

        self.fc1 = nn.Linear(784, 128)

        # 2nd hidden layer

        self.fc2 = nn.Linear(128, 64)

        # output layer

        self.fc3 = nn.Linear(64, 10)

        

        # Activation fct

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        

        x = self.fc2(x)

        x = self.relu(x)

        

        x = self.fc3(x)

        x = self.softmax(x)

        

        return x

        

model = My_Network()

model
print(model.fc1.weight)

print(model.fc1.bias)
# Set biases to all zeros

model.fc1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01

model.fc1.weight.data.normal_(std=0.01)
# Define helper function

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True





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





def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')



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