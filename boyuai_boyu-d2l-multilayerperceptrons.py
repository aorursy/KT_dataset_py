%matplotlib inline

import torch

import numpy as np

import matplotlib.pylab as plt

import sys

kaggle_path_prefix = "../input/"

sys.path.append(kaggle_path_prefix)

import d2lpytorch as d2l

print(torch.__version__)
def xyplot(x_vals, y_vals, name):

    d2l.set_figsize(figsize=(5, 2.5))

    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())

    d2l.plt.xlabel('x')

    d2l.plt.ylabel(name + '(x)')
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

y = x.relu()

xyplot(x, y, 'relu')
y.sum().backward()

xyplot(x, x.grad, 'grad of relu')
y = x.sigmoid()

xyplot(x, y, 'sigmoid')
x.grad.zero_()

y.sum().backward()

xyplot(x, x.grad, 'grad of sigmoid')
y = x.tanh()

xyplot(x, y, 'tanh')
x.grad.zero_()

y.sum().backward()

xyplot(x, x.grad, 'grad of tanh')
import torch

import numpy as np

import sys

kaggle_path_prefix = "../input/"

sys.path.append(kaggle_path_prefix)

import d2lpytorch as d2l

print(torch.__version__)
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256



W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)

b1 = torch.zeros(num_hiddens, dtype=torch.float)

W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)

b2 = torch.zeros(num_outputs, dtype=torch.float)



params = [W1, b1, W2, b2]

for param in params:

    param.requires_grad_(requires_grad=True)
def relu(X):

    return torch.max(input=X, other=torch.tensor(0.0))
def net(X):

    X = X.view((-1, num_inputs))

    H = relu(torch.matmul(X, W1) + b1)

    return torch.matmul(H, W2) + b2
loss = torch.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0

# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,

#               params=None, lr=None, optimizer=None):

#     for epoch in range(num_epochs):

#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

#         for X, y in train_iter:

#             y_hat = net(X)

#             l = loss(y_hat, y).sum()

#             

#             # 梯度清零

#             if optimizer is not None:

#                 optimizer.zero_grad()

#             elif params is not None and params[0].grad is not None:

#                 for param in params:

#                     param.grad.data.zero_()

#            

#             l.backward()

#             if optimizer is None:

#                 d2l.sgd(params, lr, batch_size)

#             else:

#                 optimizer.step()  # “softmax回归的简洁实现”一节将用到

#             

#             

#             train_l_sum += l.item()

#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()

#             n += y.shape[0]

#         test_acc = evaluate_accuracy(test_iter, net)

#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'

#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))



d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
import torch

from torch import nn

from torch.nn import init

import numpy as np

import sys

kaggle_path_prefix = "../input/"

sys.path.append(kaggle_path_prefix)

import d2lpytorch as d2l



print(torch.__version__)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

    

net = nn.Sequential(

        d2l.FlattenLayer(),

        nn.Linear(num_inputs, num_hiddens),

        nn.ReLU(),

        nn.Linear(num_hiddens, num_outputs), 

        )

    

for params in net.parameters():

    init.normal_(params, mean=0, std=0.01)
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = torch.nn.CrossEntropyLoss()



optimizer = torch.optim.SGD(net.parameters(), lr=0.5)



num_epochs = 5

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)