%load_ext autoreload

%autoreload 2

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob

import os.path as osp

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

%matplotlib inline
class MNIST(Dataset):

    """

    A customized data loader for MNIST.

    """

    def __init__(self,

                 root,

                 transform=None,

                 preload=False):

        """ Intialize the MNIST dataset

        

        Args:

            - root: root directory of the dataset

            - tranform: a custom tranform function

            - preload: if preload the dataset into memory

        """

        self.images = None

        self.labels = None

        self.filenames = []

        self.root = root

        self.transform = transform



        # read filenames

        for i in range(10):

            filenames = glob.glob(osp.join(root, str(i), '*.png'))

            for fn in filenames:

                self.filenames.append((fn, i)) # (filename, label) pair

                

        # if preload dataset into memory

        if preload:

            self._preload()

            

        self.len = len(self.filenames)

                              

    def _preload(self):

        """

        Preload dataset to memory

        """

        self.labels = []

        self.images = []

        for image_fn, label in self.filenames:            

            # load images

            image = Image.open(image_fn)

            # avoid too many opened files bug

            self.images.append(image.copy())

            image.close()

            self.labels.append(label)



    def __getitem__(self, index):

        """ Get a sample from the dataset

        """

        if self.images is not None:

            # If dataset is preloaded

            image = self.images[index]

            label = self.labels[index]

        else:

            # If on-demand data loading

            image_fn, label = self.filenames[index]

            image = Image.open(image_fn)

            

        # May use transform function to transform samples

        # e.g., random crop, whitening

        if self.transform is not None:

            image = self.transform(image)

        # return image and label

        return image, label



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len
# Create the MNIST dataset. 

# transforms.ToTensor() automatically converts PIL images to

# torch tensors with range [0, 1]

trainset = MNIST(

    root='../input/mnistpng/mnist_png/training',

    preload=True, transform=transforms.ToTensor(),

)



# Use the torch dataloader to iterate through the dataset

trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)



# load the testset

testset = MNIST(

    root='../input/mnistpng/mnist_png/testing',

    preload=True, transform=transforms.ToTensor(),

)

# Use the torch dataloader to iterate through the dataset

testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)
print(len(trainset))

print(len(testset))
# functions to show an image

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



# get some random training images

dataiter = iter(trainset_loader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % labels[j] for j in range(16)))
# Use GPU if available, otherwise stick with cpu

use_cuda = torch.cuda.is_available()

torch.manual_seed(123)

device = torch.device(cuda if use_cuda else "cpu")

print(device)


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)

        self.fc2 = nn.Linear(50, 10)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)



model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(epoch, log_interval=100):

    model.train()  # set training mode

    iteration = 0

    for ep in range(epoch):

        for batch_idx, (data, target) in enumerate(trainset_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = F.nll_loss(output, target)

            loss.backward()

            optimizer.step()

            if iteration % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                    ep, batch_idx * len(data), len(trainset_loader.dataset),

                    100. * batch_idx / len(trainset_loader), loss.item()))

            iteration += 1

        test()


def test():

    model.eval()  # set evaluation mode

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in testset_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(testset_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(testset_loader.dataset),

        100. * correct / len(testset_loader.dataset)))
train(5) 


def save_checkpoint(checkpoint_path, model, optimizer):

    state = {'state_dict': model.state_dict(),

             'optimizer' : optimizer.state_dict()}

    torch.save(state, checkpoint_path)

    print('model saved to %s' % checkpoint_path)

    

def load_checkpoint(checkpoint_path, model, optimizer):

    state = torch.load(checkpoint_path)

    model.load_state_dict(state['state_dict'])

    optimizer.load_state_dict(state['optimizer'])

    print('model loaded from %s' % checkpoint_path)
model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

test()
def train_save(epoch, save_interval, log_interval=100):

    model.train()  # set training mode

    iteration = 0

    for ep in range(epoch):

        for batch_idx, (data, target) in enumerate(trainset_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = F.nll_loss(output, target)

            loss.backward()

            optimizer.step()

            if iteration % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                    ep, batch_idx * len(data), len(trainset_loader.dataset),

                    100. * batch_idx / len(trainset_loader), loss.item()))

            if iteration % save_interval == 0 and iteration > 0:

                save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)

            iteration += 1

        test()

    

    # save the final model

    save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)


train_save(5, 500, 100)


# create a new model

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# load from the final checkpoint

load_checkpoint('mnist-4690.pth', model, optimizer)

# should give you the final model accuracy

test()


# What's in a state dict?

print(model.state_dict().keys())
checkpoint = torch.load('mnist-4690.pth')

states_to_load = {}

for name, param in checkpoint['state_dict'].items():

    if name.startswith('conv'):

        states_to_load[name] = param



# Construct a new state dict in which the layers we want

# to import from the checkpoint is update with the parameters

# from the checkpoint

model_state = model.state_dict()

model_state.update(states_to_load)

        

model = Net().to(device)

model.load_state_dict(model_state)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train(1)  # training 1 epoch will get you to 93%!


class SmallNet(nn.Module):

    def __init__(self):

        super(SmallNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 10)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = self.fc1(x)

        return F.log_softmax(x, dim=1)



model = SmallNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
checkpoint = torch.load('mnist-4690.pth')

states_to_load = {}

for name, param in checkpoint['state_dict'].items():

    if name.startswith('conv'):

        states_to_load[name] = param



# Construct a new state dict in which the layers we want

# to import from the checkpoint is update with the parameters

# from the checkpoint

model_state = model.state_dict()

model_state.update(states_to_load)

        

model.load_state_dict(model_state)
train(1)  # training 1 epoch will get you to 95%!