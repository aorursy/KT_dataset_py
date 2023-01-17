import torch

import time



n = 1000

a = torch.ones(n)

b = torch.ones(n)

# define the Timer class to record time 

class Timer(object):

    """Record multiple running times."""

    def __init__(self):

        self.times = []

        self.start()



    def start(self):

        # Start the timer

        self.start_time = time.time()



    def stop(self):

        # Stop the timer and record the time in a list

        self.times.append(time.time() - self.start_time)

        return self.times[-1]



    def avg(self):

        # Return the average time

        return sum(self.times)/len(self.times)



    def sum(self):

        # Return the sum of time

        return sum(self.times)
timer = Timer()

c = torch.zeros(n)

for i in range(n):

    c[i] = a[i] + b[i]

'%.5f sec' % timer.stop()
timer.start()

d = a + b

'%.5f sec' % timer.stop()
%matplotlib inline

import torch

from IPython import display

from matplotlib import pyplot as plt

import numpy as np

import random



print(torch.__version__)


num_inputs = 2

num_examples = 1000

true_w = [2, -3.4]

true_b = 4.2

features = torch.randn(num_examples, num_inputs,

                      dtype=torch.float32)

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),

                       dtype=torch.float32)
def use_svg_display():

    # display in vector graph

    display.set_matplotlib_formats('svg')



def set_figsize(figsize=(3.5, 2.5)):

    use_svg_display()

    # set the size of figure

    plt.rcParams['figure.figsize'] = figsize



set_figsize()

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
def data_iter(batch_size, features, labels):

    num_examples = len(features)

    indices = list(range(num_examples))

    random.shuffle(indices)  # random read 10 samples

    for i in range(0, num_examples, batch_size):

        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch

        yield  features.index_select(0, j), labels.index_select(0, j)
batch_size = 10



for X, y in data_iter(batch_size, features, labels):

    print(X, '\n', y)

    break
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)

b = torch.zeros(1, dtype=torch.float32)



w.requires_grad_(requires_grad=True)

b.requires_grad_(requires_grad=True)
def linreg(X, w, b):

    return torch.mm(X, w) + b
def squared_loss(y_hat, y): 

    return (y_hat - y.view(y_hat.size())) ** 2 / 2
def sgd(params, lr, batch_size): 

    for param in params:

        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track
lr = 0.03

num_epochs = 5

net = linreg

loss = squared_loss



for epoch in range(num_epochs):  # training repeats num_epochs times

    # in each epoch, all the samples in dataset will be used once

    # X is the feature and y is the label of a batch sample

    for X, y in data_iter(batch_size, features, labels):

        l = loss(net(X, w, b), y).sum()  # l is the loss of the batch sample

        l.backward()  # calculate the gradient of batch sample loss 

        sgd([w, b], lr, batch_size)  # using small batch random gradient descent to iter model parameters

        

        # reset parameter gradient

        w.grad.data.zero_()

        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)

    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
w, true_w, b, true_b
import torch

from torch import nn

import numpy as np

torch.manual_seed(1)



print(torch.__version__)

torch.set_default_tensor_type('torch.FloatTensor')
num_inputs = 2

num_examples = 1000

true_w = [2, -3.4]

true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
import torch.utils.data as Data



batch_size = 10



# combine featues and labels of dataset

dataset = Data.TensorDataset(features, labels)



# put dataset into DataLoader

data_iter = Data.DataLoader(

    dataset=dataset,            # torch TensorDataset format

    batch_size=batch_size,      # mini batch size

    shuffle=True,               # whether shuffle the data or not

    num_workers=2,              # read data in multithreading

)
for X, y in data_iter:

    print(X, '\n', y)

    break
class LinearNet(nn.Module):

    def __init__(self, n_feature):

        super(LinearNet, self).__init__()      # call father function to init 

        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`



    def forward(self, x):

        y = self.linear(x)

        return y

    

net = LinearNet(num_inputs)

print(net)


# method one

net = nn.Sequential(

    nn.Linear(num_inputs, 1)

    # other layers can be added here

    )



# method two

net = nn.Sequential()

net.add_module('linear', nn.Linear(num_inputs, 1))

# net.add_module ......



# method three

from collections import OrderedDict

net = nn.Sequential(OrderedDict([

          ('linear', nn.Linear(num_inputs, 1))

          # ......

        ]))



print(net)

print(net[0])
from torch.nn import init



init.normal_(net[0].weight, mean=0.0, std=0.01)

init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly
for param in net.parameters():

    print(param)
loss = nn.MSELoss()    # nn built-in squared loss function

                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`
import torch.optim as optim



optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function

print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
num_epochs = 3

for epoch in range(1, num_epochs + 1):

    for X, y in data_iter:

        output = net(X)

        l = loss(output, y.view(-1, 1))

        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()

        l.backward()

        optimizer.step()

    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net[0]

print(true_w, dense.weight.data)

print(true_b, dense.bias.data)