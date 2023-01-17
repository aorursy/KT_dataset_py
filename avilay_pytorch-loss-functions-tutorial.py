import numpy as np

import torch as t

import torch.nn as nn

import torch.nn.functional as F
loss_fn = nn.L1Loss(reduction='mean')

y_hat = t.tensor([1., 2., 3.])

y = t.tensor([1.2, 2.2, 3.2])

loss = loss_fn(y_hat, y)

print(loss)
loss_fn = nn.MSELoss(reduction='mean')

y_hat = t.tensor([1., 2., 3.])

y = t.tensor([1.2, 2.2, 3.2])

loss = loss_fn(y_hat, y)

print(loss)
# Minibatch size m = 2

# Number of classes c = 3

l = t.tensor([[1., 2., 3.],

              [4., 5., 6.]])

y = t.tensor([0, 1])



# This will select 1 from the first row of the l matrix, and 5 from its second row. Then it'll negate the selected values,

# and return the average of these two numbers which is equal to -3

loss_fn = nn.NLLLoss(reduction='mean')

loss_fn(l, y)
# Minibatch size m = 2

# Number of classes k = 3

h = t.tensor([[1., 2., 3.],

              [4., 5., 6.]])

y = t.tensor([0, 1])



loss_fn = nn.CrossEntropyLoss(reduction='elementwise_mean')

loss_fn(h, y)
# Lets do this by hand

p = F.softmax(h, dim=1)

l = t.log(p)

print(p)

print(l)



# Log likelihood of the first row is the 0th element because y[0] = 0

print(l[0, 0])



# Log likelihood of the second row is the 1st element because y[1] = 1

print(l[1, 1])



# The final loss is the average of the negative log likelihoods

loss = ((-l[0, 0]) + (-l[1, 1])) / 2

print(loss)
y = t.tensor([1., 0., 1.])

p = t.tensor([0.9, 0.7, 0.8])

loss_fn = nn.BCELoss(reduction='elementwise_mean')

loss = loss_fn(p, y)

print(loss)



# By hand

l = [None, None, None]

l[0] = np.log(0.9)

l[1] = np.log(0.3)

l[2] = np.log(0.8)

loss = - sum(l)/3

print(loss)
y = t.tensor([[1., 1., 0.],

              [0., 1., 1.]])

p = t.tensor([[0.9, 0.8, 0.2],

              [0.7, 0.8, 0.6]])

loss_fn = nn.BCELoss(reduction='elementwise_mean')

loss = loss_fn(p, y)

print(loss)



# By hand

l = [None, None]

l[0] = (np.log(0.9) + np.log(0.8) + np.log(0.8))/3

l[1] = (np.log(0.3) + np.log(0.8) + np.log(0.6))/3

loss = - sum(l) / 2

print(loss)