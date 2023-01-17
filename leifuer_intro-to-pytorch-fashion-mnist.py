import torch

from torchvision import datasets, transforms

import helper



# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data

trainset = datasets.FashionMNIST('/F_MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)



# Download and load the test data

testset = datasets.FashionMNIST('/F_MNIST_data/', download=True, train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
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

    img = img.cpu()

    ps = ps.cpu() # convert cuda to cpu for numpy 

    

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
image, label = next(iter(trainloader))

imshow(image[0,:]);
# TODO: Define your network architecture here

import torch

from torch import nn

# import torch.nn.functional as F





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = nn.Sequential(

                      nn.Linear(784, 256),

                      nn.ReLU(),

                      nn.Linear(256, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1)

                      ).to(device)

# TODO: Create the network, define the criterion and optimizer

from torch import optim



criterion = nn.NLLLoss()



optimizer = optim.Adam(model.parameters(), lr=0.03)
# TODO: Train the network here

epochs = 5

for e in range(1, epochs+1):

    running_loss = 0

    for image, label in trainloader:

        image, label = image.to(device), label.to(device)

        image = image.view(image.shape[0], -1)

        

        optimizer.zero_grad()

        

        output = model(image)

        

        loss = criterion(output, label)

        

        loss.backward()

        

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        print(f'Training loss: {loss} Epoch: {e}/{epochs}')
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



# import helper



# Test out your network!



dataiter = iter(testloader)

images, labels = dataiter.next()

images, labels = images.to(device), labels.to(device)

img = images[0]

# Convert 2D image to 1D vector

img = img.resize_(1, 784)



# TODO: Calculate the class probabilities (softmax) for img

with torch.no_grad():

    logits = model(img)

    

ps = torch.exp(logits)



# Plot the image and probabilities

view_classify(img.resize_(1, 28, 28), ps, version='Fashion')

print(ps.max())