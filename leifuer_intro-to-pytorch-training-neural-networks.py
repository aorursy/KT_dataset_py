import torch

from torch import nn

import torch.nn.functional as F

from torchvision import datasets, transforms



# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize((0.5,), (0.5,)),

                              ])

# Download and load the training data

trainset = datasets.MNIST('/MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Build a feed-forward network

model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10))

                      ####!!!

                      # there is no softmax here because CrossEntropyLoss criterion

                      # will do it together with loss calculation. 



# Define the loss

criterion = nn.CrossEntropyLoss()



# Get our data

images, labels = next(iter(trainloader))

# Flatten images

images = images.view(images.shape[0], -1)



# Forward pass, get our logits

logits = model(images)

# Calculate the loss with the logits and the labels

loss = criterion(logits, labels)



print(loss)
# TODO: Build a feed-forward network

model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1)

                      )



# TODO: Define the loss

criterion = nn.NLLLoss() # Since we define Log Softmax output, we will use NLLLoss instead of Cross Entro



### Run this to check your work

# Get our data

images, labels = next(iter(trainloader))

# Flatten images

images = images.view(images.shape[0], -1)



# Forward pass, get our logits

logits = model(images)

# Calculate the loss with the logits and the labels

loss = criterion(logits, labels)



print(loss)
x = torch.randn(2,2, requires_grad=True)

print(x)
y = x**2

print(y)
## grad_fn shows the function that generated this variable

print(y.grad_fn)
z = y.mean()

print(z)
print(x.grad)
z.backward()

print(x.grad)

print(x/2)
# Build a feed-forward network

model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)



logits = model(images)

loss = criterion(logits, labels)
print('Before backward pass: \n', model[0].weight.grad)



loss.backward()



print('After backward pass: \n', model[0].weight.grad)
from torch import optim



# Optimizers require the parameters to optimize and a learning rate

optimizer = optim.SGD(model.parameters(), lr=0.01)
print('Initial weights - ', model[0].weight)



images, labels = next(iter(trainloader))

images.resize_(64, 784)



# Clear the gradients, do this because gradients are accumulated

optimizer.zero_grad()



# Forward pass, then backward pass, then update weights

output = model(images)

loss = criterion(output, labels)

loss.backward()

print('Gradient -', model[0].weight.grad)
# Take an update step and few the new weights

optimizer.step()

print('Updated weights - ', model[0].weight)
## Your solution here



model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.003)



epochs = 5

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        # Flatten MNIST images into a 784 long vector

        images = images.view(images.shape[0], -1)

    

        # TODO: Training pass

        optimizer.zero_grad()

        output = model(images)

                       

        loss = criterion(output, labels)

        

        loss.backward()

        

        optimizer.step()

        

        running_loss += loss.item()

    else:

        print(f"Training loss: {running_loss/len(trainloader)}")
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
%matplotlib inline

# import helper



images, labels = next(iter(trainloader))



img = images[0].view(1, 784)

# Turn off gradients to speed up this part

with torch.no_grad():

    logps = model(img)



# Output of the network are log-probabilities, need to take exponential for probabilities

ps = torch.exp(logps)

# helper.view_classify(img.view(1, 28, 28), ps)

view_classify(img.view(1, 28, 28), ps)