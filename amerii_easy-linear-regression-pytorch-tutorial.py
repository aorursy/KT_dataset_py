import numpy as np

import torch

import matplotlib.pyplot as plt

import torch.nn as nn
# number of points

N = 20



# Generates random numbers between -10 and 10

X = np.random.random(N)*20 - 10



# Generates random numbers between -2.5 and 2.5

noise = np.random.random(N)*5 - 2.5



# True slope of the line

m = 0.25



# True value of the y-intercept

c = 3



# True equation of the line

Y_true = m*X + c



# Equation of the line with noise introduced

Y = Y_true + noise
plt.scatter(X,Y_true);
plt.scatter(X,Y);
# create a model with one input and one output

model = nn.Linear(1,1)
# Define Loss and Optimizer 

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)
display(X.shape)

display(Y.shape)
# Reshape our data into appropriate dimesnions so that we can feed it into PyTorch

X = X.reshape(N,1)

Y = Y.reshape(N,1)

print(f'X shape: {X.shape}')

print(f'Y shape: {Y.shape}')
# Check the data type of your the values inside your numpy array

X.dtype
# Change data type from float 64, which is the default datatype of numpy array, to float32, which is the default datatype of a torch tensor

X = X.astype('float32')

Y = Y.astype('float32')

print(f'data type of X: {X.dtype}')

print(f'data type of Y: {Y.dtype}')
# Change data from numpy format to torch tensor

inputs = torch.from_numpy(X)

targets = torch.from_numpy(Y)

print(f'type of X: {type(X)}')

print(f'type of inputs: {type(inputs)}')
# just to remeber what we are trying to map

# given a certain input we want to predict the corresponding output

for i,t in zip(inputs,targets):

    print('expected input          expected target')

    print(f'{i.item():.4f}                        {t.item():.4f}')

    print()
epochs = 300

losses = []



for i in range(epochs):

    # reset the gradients parameter

    optimizer.zero_grad()

    

    # forward propagate

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    

    # append the loss to list of losses

    losses.append(loss.item())

    

    # back propagate (calculate the gradients)

    loss.backward()

    

    # apply gradient descent

    # performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule

    optimizer.step()

    

    print(f'Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}')

    
plt.plot(losses[50:100]);
# Plot the graph

predicted = model(inputs).detach().numpy()

plt.scatter(X, Y, label='Original data')

plt.plot(X, predicted, label='Fitted line')

plt.legend()

plt.show()
# Compare obtained weight and bias with true slope and y-intercept



w = model.weight.data.numpy()

b = model.bias.data.numpy()

print(f'True Slope: {m}')

print(f'Obtained Weight: {w.squeeze():.3F}')

print()

print(f'True y-intercept: {c}')

print(f'Obtained y-intercept: {b.squeeze():.3F}')