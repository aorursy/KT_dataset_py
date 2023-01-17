import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random

# ================================================================== #
#                    0. Reproducibility                              #
# ================================================================== #
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)
np.random.seed(17)
random.seed(17)

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
# Esse seria o passo foward
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 
# ================================================================== #
#                    2. Basic autograd example 2                     #  
#                         Exemplo de Regressão                       #
# ================================================================== #
# Dados de Entrada
x = torch.randn(10, 3)
x
#simulando, então, 10 dados e três features
x.shape
y = torch.randn(10, 1)
y
# Build a fully connected layer.
#Como ela é linear, não precisamos de uma função de ativação
linear = nn.Linear(3, 1)
print ('w: ', linear.weight)
print ('b: ', linear.bias)
# Definição da nossa função de erro
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)
# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())
# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
optimizer.zero_grad()
# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
#Ajuste dos pesos pelo gradiente descendente
optimizer.step()

# Print out the loss after 2-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 2 step optimization: ', loss.item())
x = torch.randn(10, 3)
y = torch.randn(10, 1)
linear = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
pred = linear(x)
pred
sigmoid = nn.Sigmoid()
sigmoid(pred)