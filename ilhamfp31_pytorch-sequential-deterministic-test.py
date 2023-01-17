import pandas as pd

import os

import random

import numpy as np

import torch

import torchvision

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import matplotlib.pyplot as plt

%matplotlib inline



train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')

    

def set_seed(seed=1):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True

    

print("PyTorch Version: {}".format(torch.__version__))

print("NumPy Version: {}".format(np.__version__))
# constants

RANDOM_SEED = 1

n_epochs = 3

batch_size_train = 64

batch_size_test = 1000

learning_rate = 0.01

momentum = 0.5

log_interval = 10
train_loader = torch.utils.data.DataLoader(

  torchvision.datasets.MNIST('/files/', train=True, download=True,

                             transform=torchvision.transforms.Compose([

                               torchvision.transforms.ToTensor(),

                               torchvision.transforms.Normalize(

                                 (0.1307,), (0.3081,))

                             ])),

  batch_size=batch_size_train)



test_loader = torch.utils.data.DataLoader(

  torchvision.datasets.MNIST('/files/', train=False, download=True,

                             transform=torchvision.transforms.Compose([

                               torchvision.transforms.ToTensor(),

                               torchvision.transforms.Normalize(

                                 (0.1307,), (0.3081,))

                             ])),

  batch_size=batch_size_test, shuffle=True)
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

        return F.log_softmax(x)
def train(network, epoch):

  network.train()

  for batch_idx, (data, target) in enumerate(train_loader):

    optimizer.zero_grad()

    output = network(data)

    loss = F.nll_loss(output, target)

    loss.backward()

    optimizer.step()

    

def test(network):

  network.eval()

  test_loss = 0

  correct = 0

  with torch.no_grad():

    for data, target in test_loader:

      output = network(data)

      test_loss += F.nll_loss(output, target, size_average=False).item()

      pred = output.data.max(1, keepdim=True)[1]

      correct += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(

    test_loss, correct, len(test_loader.dataset),

    100. * correct / len(test_loader.dataset)))
# Save a random initialization

set_seed(seed=RANDOM_SEED)

torch.save(Net(), 'init.pth')



for counter in range(1, 6):

    print("Train #{}".format(counter))

    network = torch.load('init.pth')

    

    print(network.conv1.weight[0])

    

    optimizer = optim.SGD(network.parameters(), lr=learning_rate,

                          momentum=momentum)

    

    for epoch in range(1, n_epochs + 1):

      train(network, epoch)

    

    test(network)