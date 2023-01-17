import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#Criate data for a y = 2x + 1 equation with noise
x_train = np.arange(0,10,.1).reshape(-1,1)
y_train = (2 * x_train + 1) + (np.random.randn(len(x_train))).reshape(-1,1)
#Turn the numpy arrays to float32 to avoid an error when converting to tensors
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
#Create the 1 layer NN
class LinearRegressionModel(nn.Module):

    def __init__(self, input_size, output_size):

        super(LinearRegressionModel, self).__init__()
    
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out
#1 input (x) 1 outpuy (y)
input_dim = 1
output_dim = 1
#Create an instance of the NN
model = LinearRegressionModel(input_dim, output_dim)
#Define the loss function to MSE
criterion = nn.MSELoss()
#Learning rate and optimizer (SGD)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#Optimization loop
epochs = 100
for epoch in range(epochs):
    epoch += 1

    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    #Clean the gradients calculated onthe last loop
    optimizer.zero_grad()
    #Apply the forward
    outputs = model.forward(inputs)
    #Calculate the loss between outputs and labels
    loss = criterion(outputs, labels)
    #Calculate the gradients
    loss.backward()
    #Apply the parameters values for the weights
    optimizer.step()
    #Print the loss value every 10 epochs
    if epoch % 10 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.data))
#plot the model v the data
prev = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.scatter(x_train, y_train)
plt.plot(x_train, prev, c='r')
plt.show()