import torch

from torchvision import datasets, transforms



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
from torch import nn, optim

import torch.nn.functional as F



class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)

        

    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)

        

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x
toto = torch.randn(12,1,3,4)

print(toto.shape)

toto.view(toto.shape[0],-1).shape
model = Classifier()



images, labels = next(iter(testloader))

print(images.shape)

# Get the class probabilities

ps = torch.exp(model(images))

# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples

print(ps.shape)
top_p, top_class = ps.topk(1, dim=1)

# Look at the most likely classes for the first 10 examples

print(top_p[:10])

print(top_class[:10,:])
equals = top_class == labels

print(equals.shape)
equals = top_class == labels.view(*top_class.shape)

print(equals.shape)
accuracy = torch.mean(equals)
accuracy = torch.mean(equals.type(torch.FloatTensor))

print(f'Accuracy: {accuracy.item()*100}%')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier()



model.to(device) # Move to GPU



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 30

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        # Move to GPU for acceleration

        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        ## TODO: Implement the validation pass and print out the validation accuracy

        with torch.no_grad():

            for images, labels in testloader:

                images, labels = images.to(device), labels.to(device)

                

                ps = torch.exp(model(images))



                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy = torch.mean(equals.type(torch.FloatTensor))

        print(f'Accuracy: {accuracy.item()*100}%')
## TODO: Define your model with dropout added

class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)

        

        # Dropout

        self.dropout = nn.Dropout(p=0.2)

        

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        

        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x
## TODO: Train your model with dropout, and monitor the training progress with the validation loss and accuracy

model = Classifier()

model.to(device)



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 30



train_losses, test_losses = [], []



for e in range(1, epochs+1):

    running_loss = 0

    for images, labels in trainloader:

        # Move to GPU

        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        train_losses.append(running_loss/len(trainloader))

    else:

        test_loss, accuracy = 0, 0

        

        with torch.no_grad():

            

            model.eval()

            for images, labels in testloader:

                # Move to GPU

                images, labels = images.to(device), labels.to(device)

                

                logits = model(images)

                test_loss += criterion(logits, labels)

                

                ps = torch.exp(logits)

                

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        test_losses.append(test_loss/len(testloader))

        model.train()

        

        print(f'Epoch: {e}/{epochs}',

              f'Training loss: {running_loss/len(trainloader)}',

              f'Test loss: {test_loss/len(testloader)}',

              f'Accuracy: {accuracy/len(testloader)}')

        
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
# Import helper module (should be in the repo)

import helper



# Test out your network!



model.eval()



# Move back to CPU

model.cpu()



dataiter = iter(testloader)

images, labels = dataiter.next()

img = images[0]

# Convert 2D image to 1D vector

img = img.view(1, 784)



# Calculate the class probabilities (softmax) for img

with torch.no_grad():

    output = model.forward(img)



ps = torch.exp(output)



# Plot the image and probabilities

# helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')

view_classify(img.view(1, 28, 28), ps, version='Fashion')

print(ps.max())