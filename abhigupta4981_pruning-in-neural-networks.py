import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import torch.nn.functional as F

from tqdm import tqdm

from pylab import rcParams

import copy

import time

%matplotlib inline
train_ds = datasets.MNIST(root='/tmp', train=True, download=True, transform=transforms.ToTensor())

test_ds = datasets.MNIST(root='/tmp', train=False, download=True, transform=transforms.ToTensor())
batch_size = 128
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 1000)

        self.fc2 = nn.Linear(1000, 1000)

        self.fc3 = nn.Linear(1000, 500)

        self.fc4 = nn.Linear(500, 200)

        self.fc5 = nn.Linear(200, 10)

    def forward(self, x):

        x = x.view(-1, 28*28)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        out = self.fc5(x)

        return out
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()
num_epochs = 25
training_losses = []

training_accuracies = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.

    correct = 0

    total = 0

    for batch_idx, (data, target) in tqdm(enumerate(train_dl)):

        data = data.to(device)

        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        _, predicted = torch.max(output.data, 1)

        total+=target.size(0)

        correct+=(predicted==target).sum().item()

    training_losses.append(running_loss/total)

    training_accuracies.append(correct/total)
plt.figure(1, figsize=(15, 3))

plt.subplot(121)

plt.plot(training_losses)

plt.xlabel('Epoch')

plt.ylabel('Training Loss')

plt.subplot(122)

plt.plot(training_accuracies)

plt.xlabel('Epoch')

plt.ylabel('Training Accuracy')
def test(model, test_dl):

    correct = 0

    total = 0

    with torch.no_grad():

        for data, target in tqdm(test_dl):

            data = data.to(device)

            target = target.to(device)

            output = model(data)

            _, predicted = torch.max(output.data, 1)

            total+=target.size(0)

            correct+=(predicted==target).sum().item()

    return correct/total
original_accuracy = test(model, test_dl)

original_accuracy
def weight_prune(model, pruning_percentage, test_dl):

    model1 = copy.deepcopy(model)

    length = len(list(model1.parameters()))

    for i, param in enumerate(model1.parameters()):

        if len(param.size())!=1 and i<length-2:

            weight = param.detach().cpu().numpy()

            weight[np.abs(weight)<np.percentile(np.abs(weight), pruning_percentage)] = 0

            weight = torch.from_numpy(weight).to(device)

            param.data = weight

    return test(model1, test_dl)
pruning_percent = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]
accuracy_weight_pruning = []

for percent in pruning_percent:

    accuracy_weight_pruning.append(weight_prune(model, percent, test_dl))
rcParams['figure.figsize'] = 12, 8

plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r',

         pruning_percent, accuracy_weight_pruning, 'b')

plt.grid()

plt.legend([['Original Accuracy'], 

            ['Accuracy with weight pruning']],

           loc='lower left', fontsize='xx-large')

plt.xlabel('Pruning Percentage', fontsize='xx-large')

plt.ylabel('Accuracy', fontsize='xx-large')

plt.xticks(pruning_percent)

plt.yticks(np.arange(0, 1.05, 0.05))

plt.show()
def neuron_pruning(model, pruning_percentage, test_dl):

    model1 = copy.deepcopy(model)

    length = len(list(model1.parameters()))

    for i, param in enumerate(model1.parameters()):

        if len(param.size())!=1 and i<length-2:

            weight = param.detach().cpu().numpy()

            norm = np.linalg.norm(weight, axis=0)

            weight[:, np.argwhere(norm<np.percentile(norm, pruning_percentage))] = 0

            weight = torch.from_numpy(weight).to(device)

            param.data = weight

    return test(model1, test_dl)
accuracy_neuron_pruning = []

for percent in pruning_percent:

    accuracy_neuron_pruning.append(neuron_pruning(model, percent, test_dl))
rcParams['figure.figsize'] = 12, 8

plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r', pruning_percent, accuracy_neuron_pruning, 'b')

plt.grid()

plt.legend([['Original Accuracy'], ['Accuracy with neuron pruning']], loc='lower left', fontsize='xx-large')

plt.xlabel('Pruning Percentage', fontsize='xx-large')

plt.ylabel('Accuracy', fontsize='xx-large')

plt.xticks(pruning_percent)

plt.yticks(np.arange(0, 1.05, 0.05))

plt.show()
rcParams['figure.figsize'] = 12, 8

plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r',

         pruning_percent, accuracy_weight_pruning, 'b',

         pruning_percent, accuracy_neuron_pruning, 'g')

plt.grid()

plt.legend([['Original Accuracy'],

            ['Accuracy with weight pruning'], 

            ['Accuracy with neuron pruning']], 

           loc='lower left', fontsize='xx-large')

plt.xlabel('Pruning Percentage', fontsize='xx-large')

plt.ylabel('Accuracy', fontsize='xx-large')

plt.xticks(pruning_percent)

plt.yticks(np.arange(0, 1.05, 0.05))

plt.show()