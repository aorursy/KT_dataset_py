# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
from torch import nn
from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(train.shape)
train.head()
X = train.loc[:, train.columns != 'label']
print(type(X), X.shape)
X = X.values / 255  # [0, 255] -> [0, 1]
print(type(X), X.shape)
y = train.label
print(type(y), y.shape)
y = y.values
print(type(y), y.shape)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)

print(type(train_X), train_X.shape)
print(type(test_X), test_X.shape)
print(type(train_y), train_y.shape)
print(type(test_y), test_y.shape)

train_X = torch.from_numpy(train_X)
test_X = torch.from_numpy(test_X)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

print(type(train_X), train_X.shape)
print(type(test_X), test_X.shape)
print(type(train_y), train_y.shape)
print(type(test_y), test_y.shape)

train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(train_loader.dataset.tensors[0].shape)
print(train_loader.dataset.tensors[1].shape, train_loader.dataset.tensors[1][0])
input_size = 28 * 28
hidden_size = [128, 64]
output_size = 10

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        # Hidden layer 1 to HL2 linear transformation
        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        # HL2 to output linear transformation
        self.layer3 = nn.Linear(hidden_size[1], output_size)

        # RELU = max(0, x)
        self.relu = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))

        # Output layer with LogSoftmax activation
        out = self.LogSoftmax(self.layer3(out))
        return out
model = NeuralNet(input_size, hidden_size, output_size)

# for idx, m in enumerate(model.modules()):  # named_modules()
#     print(idx, '->', m, '\n')

# for param in model.parameters():  # named_parameters()
#     print(param)

# model.requires_grad_(requires_grad=True)

# Returns a dictionary containing a whole state of the module.
# model.state_dict().keys()  # makes an OrderedDict from weights and biases

print(model)
lossFunction = nn.NLLLoss()
lossFunction
optimSGD = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimSGD
print(len(train_loader.dataset.tensors))
a, b = train_loader.dataset.tensors
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break
num_epochs = 101
for epoch in range(num_epochs):
    loss_ = 0
    for images, labels in train_loader:
        # Flatten the input images of [28,28] to [1,784]
        # images = images.reshape(-1, 784)

        # Forward Pass
        output = model(images.type(torch.float))
        # Loss at each iteration by comparing to target(label)
        loss = lossFunction(output, labels)

        # Backpropagating gradient of loss
        # Clears the gradients of all optimized torch.Tensor's.
        optimSGD.zero_grad()
        loss.backward()

        # Performs a single optimization step - updating parameters(weights and bias)
        optimSGD.step()

        loss_ += loss.item()

    if (epoch % 10 == 0):
        print("Epoch {}, Training loss: {}".format(epoch, loss_ / len(train_loader)))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784)
        out = model(images.type(torch.float))
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Testing accuracy: {:.2f} %'.format(100 * correct / total))
# Define optimizers with different hyperparameters values

optimizers = {
    SGD: [
        {
            'lr': 0.001,
            'momentum': 0.0,
            'nesterov': False,
        },
        {
            'lr': 0.001,
            'momentum': 0.9,
            'nesterov': False,
        },
        {
            'lr': 0.001,
            'momentum': 0.01,
            'nesterov': True,
        },
        {
            'lr': 0.001,
            'momentum': 0.9,
            'nesterov': True,
        },
    ],
    Adam: [
        {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        {
            'lr': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
    ],
    Adagrad: [
        {
            'lr': 0.001,
            'eps': 1e-10
        },
        {
            'lr': 0.01,
            'eps': 1e-10
        },
    ],
    Adadelta: [
        {
            'lr': 0.001,
            'rho': 0.9,
            'eps': 1e-6
        },
        {
            'lr': 0.01,
            'rho': 0.9,
            'eps': 1e-6
        },
    ],
    RMSprop: [
        {
            'lr': 0.001,
            'momentum': 0.1,
            'alpha': 0.99,
            'eps': 1e-8,
            'centered': False
        },
        {
            'lr': 0.01,
            'momentum': 0.1,
            'alpha': 0.99,
            'eps': 1e-8,
            'centered': True
        },
        {
            'lr': 0.001,
            'momentum': 0.9,
            'alpha': 0.99,
            'eps': 1e-8,
            'centered': True
        }
    ]
}
# Get dict of needed hyperparameters

def get_params(_opt_type, _params, _model) -> dict:
    kwargs = dict(params=_model.parameters(), lr=_params['lr'])

    if _opt_type == SGD:
        kwargs.update(dict(momentum=_params['momentum'], nesterov=_params['nesterov']))
    elif _opt_type == Adam:
        kwargs.update(dict(betas=_params['betas'], eps=_params['eps']))
    elif _opt_type == Adagrad:
        kwargs.update(dict(eps=_params['eps']))
    elif _opt_type == RMSprop:
        kwargs.update(dict(momentum=_params['momentum'], alpha=_params['alpha'], eps=_params['eps'], centered=_params['centered']))
    elif _opt_type == Adadelta:
        kwargs.update(dict(rho=_params['rho'], eps=_params['eps']))

    return kwargs

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

def get_pretty_opt_type(_optimizer_type) -> str:
    return str(_optimizer_type).split("'")[1].split(".")[-1]


def get_opt_info(_optimizer_type, _params) -> str:
    info = 'Optimizer: {}. Params: '.format(get_pretty_opt_type(_optimizer_type))
    for key, value in _params.items():
        info += '{}={} '.format(key, value)
    return info

res = {
    'EpochCount': 31,
    'data': []
}

epoch_count = res['EpochCount']

for optimizer_type, params in optimizers.items():
    for param_set in params:
        model = NeuralNet(input_size, hidden_size, output_size)

        kwargs = get_params(optimizer_type, param_set, model)
        optimizer = optimizer_type(**kwargs)

        lossFunction = nn.NLLLoss()
        
        optimizer_info = get_opt_info(optimizer_type, param_set)
        print(optimizer_info)

        for epoch in range(epoch_count):
            loss_ = 0
            for images, labels in train_loader:
                # Forward Pass
                output = model(images.type(torch.float))
                # Loss at each iteration by comparing to target(label)
                loss = lossFunction(output, labels)

                # Backpropagating gradient of loss
                # Clears the gradients of all optimized torch.Tensor's.
                optimizer.zero_grad()
                loss.backward()

                # Performs a single optimization step - updating parameters(weights and bias)
                optimizer.step()

                loss_ += loss.item()

            print("Epoch {}, Training loss: {:.6f}".format(epoch, loss_ / len(train_loader)))


        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                out = model(images.type(torch.float))
                _, predicted = torch.max(out, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Testing accuracy: {:.3f} %\n\n'.format(100 * correct / total))







