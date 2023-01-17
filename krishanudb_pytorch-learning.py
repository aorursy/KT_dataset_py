# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
array = [[1, 2, 3], [4, 5, 6]]
first_array = np.array(array)
print("Array Type: {}".format(type(first_array)))
print("Array Shape {}:".format(first_array.shape))
print(first_array)
import torch
tensor = torch.tensor(array)
print("Array Type: {}".format(tensor.type))
print("Array Shape {}:".format(tensor.shape))
print(tensor)
print(np.ones((2, 3)))
print(torch.ones((2, 3)))
print(np.random.randn(2, 3))
print(torch.randn(2, 3))
array = np.random.rand(2, 3)
print("Type {}, Shape:{}".format(type(array), array.shape))
print(array)
tensor = torch.from_numpy(array)
print("Type {}, Shape:{}".format(type(tensor), tensor.shape))
print(tensor)
tensor = torch.ones(3, 3)
tensor2 = tensor * 4
print(tensor)
print("View: \n{}\n{}".format(tensor.view(9), tensor.view(9).shape))
print("Add: \n{}\n{}".format(torch.add(tensor, tensor2), torch.add(tensor, tensor2).shape))
print("Add 2: \n{}\n{}".format(tensor.add(tensor2), tensor.add(tensor2).shape))
print("Subtract: \n{}\n{}".format(torch.sub(tensor2, tensor), torch.sub(tensor2, tensor).shape))
print("Subtract 2: \n{}\n{}".format(tensor2.sub(tensor), tensor2.sub(tensor).shape))
print("Multiply: \n{}\n{}".format(torch.mul(tensor, tensor2), torch.mul(tensor, tensor2).shape))
print("Multiply 2: \n{}\n{}".format(tensor.mul(tensor2), tensor.mul(tensor2).shape))
tensor = torch.Tensor([1, 2, 3, 4, 5])
print("Mean {}".format(tensor.mean()))
tensor1 = torch.Tensor([1, 2, 3, 4, 5])
tensor2 = torch.tensor([1, 2, 3, 4, 5])

print("Type {}".format(tensor1.type()))
print("Type {}".format(tensor2.type()))
from torch.autograd import Variable
var = Variable(torch.ones(3), requires_grad = True)
print(var)
array = [2, 4]
tensor = torch.Tensor(array)

x = Variable(tensor, requires_grad = True)
y = x ** 2
print("y {}".format(y))
o = 1/2 * sum(y)
print("o {}".format(o))
o.backward()

print("x gradient {}".format(x.grad))
print(x.grad)
car_prices_array = [3, 4, 5, 6, 7, 8, 9]
car_prices_np = np.array(car_prices_array, dtype=np.float32)
car_prices_np = car_prices_np.reshape(-1, 1)
car_prices_tensor = Variable(torch.from_numpy(car_prices_np))
car_prices_np.shape
car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
car_sell_np = np.array(car_sell_array,dtype=np.float32)
car_sell_np = car_sell_np.reshape(-1,1)
car_sell_tensor = Variable(torch.from_numpy(car_sell_np))
car_sell_np.shape
import matplotlib.pyplot as plt
plt.scatter(car_prices_array, car_sell_array)
plt.show()
import torch
from torch.autograd import Variable
import torch.nn as nn
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

mse = nn.MSELoss()

learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_list = []

iteration_number = 1001
for iteration in range(iteration_number):
    optimizer.zero_grad()
    
    results = model(car_prices_tensor)
    
    loss = mse(results, car_sell_tensor)
    
    loss.backward()
    
    optimizer.step()
    
    loss_list.append(loss.data)
    
    if (iteration % 50 == 0):
        print("epoch {}, loss {}".format(iteration, loss.data))
    
plt.plot(range(iteration_number), loss_list)
plt.show()
predicted = model(car_prices_tensor).data.numpy()
plt.scatter(car_prices_np, predicted, c="red")
plt.scatter(car_prices_np, car_sell_np, c = "blue")
plt.show()

