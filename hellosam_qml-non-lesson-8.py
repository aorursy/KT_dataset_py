# CNN for MNIST with PyTorch. Example from: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
# Train and Test routine
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
# Load data
from scipy.io import loadmat
mnist = loadmat("/kaggle/input/mnist-original/mnist-original")
mnist_data = mnist["data"].T  # [70,000, 784]
mnist_label = mnist["label"][0]  # [70,000]

# 60000 is number of training examples
train_data = mnist_data[:60000]
train_label = mnist_label[:60000]

test_data = mnist_data[60000:]
test_label = mnist_label[60000:]

# Make a Dataset object to load data
class MNISTDataset(Dataset):
    def __init__(self, data, label, transform=None):
        # re-scaling to [0,1] by diving by 255.
        data = data.astype(np.float32)
        label = label.astype(np.int)
        self.data, self.label = data / 255., label
        self.transform = transform
        
    def __getitem__(self, i):
        sample_data, sample_label = self.data[i], int(self.label[i])
        sample_data = sample_data.reshape(28, 28)
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data, sample_label
    
    def __len__(self):
        return len(self.data)

# Configurations
lr = 0.5
batch_size = 64
device = torch.device('cuda:0')
epoch_nb = 5
log_interval = 100
gamma = 0.7
# Initializing dataset
# TODO: try not normalizing and see if that affects performance
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = MNISTDataset(train_data, train_label, transform=transform)
test_ds = MNISTDataset(test_data, test_label, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
# define model and optimizer
net = CnnNet()
net = net.to(device)  # send model to device
optimizer = optim.Adadelta(net.parameters(), lr=lr)
# Train loop
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(1, epoch_nb + 1):
    train(net, device, train_loader, optimizer, epoch, log_interval)
    test(net, device, test_loader)
    scheduler.step()

