%mkdir -p data/MNIST/processed # make directory for datasets.MNIST

%cp -r ../input/* data/MNIST/processed # copy from ../input to data/MNIST/processed



import os

# os.makedirs("input/MNIST/processed", exist_ok=True) # or try this

print(os.listdir("data/MNIST/processed"))
import torch

from torchvision import datasets, transforms



bs = 64 # batch size in every epoch



# trainning set

train_loader = torch.utils.data.DataLoader(

    datasets.MNIST(root = 'data', train=True, download=False,

                    transform=transforms.Compose([

                        transforms.ToTensor(),

                        transforms.Normalize((0.1307,), (0.3081,))

                    ])),

    batch_size=bs, shuffle=True) # shuffle set to True to have the data reshuffled at every epoch.



# test set

test_loader = torch.utils.data.DataLoader(

    datasets.MNIST(root = 'data', train=False, transform=transforms.Compose([

                        transforms.ToTensor(),

                        transforms.Normalize((0.1307,), (0.3081,))  # why using that?

                    ])),

    batch_size=bs*2, shuffle=True) # the validation set does not need backpropagation and thus takes less memory,

                                   # so we use a larger batch size.
print(len(train_loader.dataset)) # number of samples

print(len(train_loader)) # number of batches



print(len(test_loader.dataset))
import matplotlib.pyplot as plt

import numpy as np

import torchvision



# functions to show an image





def imshow(img):

    img = img * 0.3081 + 0.1307     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()





# get some random training images

dataiter = iter(train_loader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % labels[j] for j in range(4)))

print(images.size())
import torch.nn as nn

import torch.nn.functional as F



class LeNet5_like(nn.Module):

    def __init__(self):

        super(LeNet5_like, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(1, 6, 5)

        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(16*5*5, 120)

        self.fc2 = nn.Linear(120, 84)

        # NOTE! This layer is different from LeNet5: we do not use the Gaussian connections for simplicity.

        self.fc3 = nn.Linear(84, 10)

    

    # Connect layers, define activation functions

    def forward(self, x):

        # Max pooling over a (2, 2) window

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flat x for fc

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    

    def num_flat_features(self, x): # To see dimensions of layers

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features

    



model = LeNet5_like()

print(model)
import torch.optim as optim



optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

loss_f = nn.CrossEntropyLoss(reduction = 'mean')
def train(train_loader, model, optimizer, log_interval, epoch, criterion):

    model.train() # Sets the module in training mode.

    for batch_idx, (inputs, labels) in enumerate(train_loader, 0): # get the inputs. Start from index 0.

        

        # zero the parameter gradients

        optimizer.zero_grad()

        

        # forward + backward + optimize

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        # print statistics

        if batch_idx % log_interval == (log_interval-1):

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(inputs), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

            

def test(test_loader, model, criterion):

    model.eval() # Sets the module in evaluation mode.

    test_loss = 0 # loss compute by criterion

    correct = 0 # for computing accurate

    

    # `with` allows you to ensure that a resource is "cleaned up" 

    # when the code that uses it finishes running, even if exceptions are thrown.

    with torch.no_grad(): # It will reduce memory consumption for computations that would otherwise have requires_grad=True.

        for inputs, labels in test_loader:

            outputs = model(inputs)

            test_loss += criterion(outputs, labels).item() # sum up batch loss

            pred = outputs.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.view_as(pred)).sum().item()

    

    test_loss /= len(test_loader) # average on batches

    

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))
# Constants

epochs = 2 # how many epochs to train for

log_interval = 200 # how many batches to wait before logging training status

criterion = loss_f



for epoch in range(1, epochs + 1):

    train(train_loader, model, optimizer, log_interval, epoch, criterion)

    test(test_loader, model, criterion)
use_cuda = torch.cuda.is_available()

print(use_cuda)



device = torch.device("cuda" if use_cuda else "cpu")



model = LeNet5_like().to(device)



# Insert "data, target = data.to(device), target.to(device)" in train and test.