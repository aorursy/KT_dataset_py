import numpy as np

import pandas as pd

import os



import torch

import numpy as np 

from torch.nn import Linear, Sigmoid

from torch import optim, nn





np.random.seed(62)
data = pd.read_csv('/kaggle/input/apples-pears/apples_pears.csv')
data.head()
X = data.iloc[:,:2].values

Y = data['target'].values.reshape((-1, 1))



x1 = X[:, 0]

x2 = X[:, 1]
lr = 0.001        # learning rate

n_epochs = 10000  # number epochs



def sigmoid(x):

    return 1 / (1 + np.exp(-x))
w1 = np.random.randn(1)

w2 = np.random.randn(1)

w0 = np.random.randn(1)



idx = np.arange(1000)

np.random.shuffle(idx)

x1, x2, y = x1[idx], x2[idx], Y[idx]



for epoch in range(n_epochs):

    i = np.random.randint(0, 999)

    yhat = w1 * x1[i] + w2 * x2[i] + w0

    

    w1_grad = -((y[i] - sigmoid(yhat)) * x1[i])

    w2_grad = -((y[i] - sigmoid(yhat)) * x2[i])

    w0_grad = -(y[i] - sigmoid(yhat))

    

    w1 -= w1_grad * lr

    w2 -= w2_grad * lr

    w0 -= w0_grad * lr
print(w0, w1, w2)
i = 0

correct = 0

incorrect = 0



for item in y:

  if (np.around(x1[i] * w1 + x2[i] * w2 + w0) == item):

    correct += 1

  else:

    incorrect += 1

  i = i + 1



print(correct, incorrect)
X = torch.FloatTensor(data.iloc[:,:2].values)  

y = torch.FloatTensor(data['target'].values.reshape((-1, 1)))
def make_train_step(model, loss_fn, optimizer):

    def train_step(x, y):

        model.train()

        yhat = model(x)

        loss = loss_fn(yhat, y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        return loss.item()

    return train_step
neuron = torch.nn.Sequential(

    Linear(2, out_features=1),

    Sigmoid()

)

print(neuron.state_dict())
lr = 0.001

n_epochs = 10000



loss_fn = nn.MSELoss(reduction="mean")

optimizer = optim.SGD(neuron.parameters(), lr=lr)

train_step = make_train_step(neuron, loss_fn, optimizer)



for epoch in range(n_epochs):

    loss = train_step(X, y)
print(neuron.state_dict())

print(loss)