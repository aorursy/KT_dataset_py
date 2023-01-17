import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))
import os

print(os.listdir("../working"))

import pandas as pd

pd.DataFrame([1,2,3,4]).to_csv("../working/test.csv")

print(os.listdir("../working"))
import torch
x = torch.empty(5, 3)

print(x)
x = torch.rand(5, 3)

print(x)
x = torch.zeros(5, 3, dtype=torch.long)

print(x)

x = torch.tensor([5.5, 3])

print(x)
x = x.new_ones(5, 3, dtype=torch.double)      

print(x)
x = torch.randn_like(x, dtype=torch.float)    

print(x) 
print(x.size())
y = torch.rand(5, 3)

print(x + y)
y.view(-1,5).mm(x)
result = torch.empty(5, 3)

torch.add(x, y, out=result)

print(result)
y.add_(x)

print(y)
print(x[:, 1])
x = torch.randn(4, 4)

y = x.view(16)

z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

print(x.size(), y.size(), z.size())
x = torch.randn(1)

print(x)

print(x.item())
a = torch.ones(5)

print(a)
b = a.numpy()

print(b)
a.add_(1)

print(a)

print(b)


import numpy as np

a = np.ones(5)

b = torch.from_numpy(a)

np.add(a, 1, out=a)

print(a)

print(b)
torch.cuda.is_available()
device = torch.device("cuda")          

y = torch.ones_like(x, device=device)  

x = x.to(device)                       

z = x + y

print(z)

print(z.to("cpu", torch.double))
import numpy as np

import time



# N is batch size; D_in is input dimension;

# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 64, 1000, 100, 10



# Create random input and output data

x = np.random.randn(N, D_in)

y = np.random.randn(N, D_out)



# Randomly initialize weights

w1 = np.random.randn(D_in, H)

w2 = np.random.randn(H, D_out)



start_time=time.time()

learning_rate = 1e-6

for t in range(1):

    # Forward pass: compute predicted y

    h = x.dot(w1)

    h_relu = np.maximum(h, 0)

    y_pred = h_relu.dot(w2)



    # Compute and print loss

    loss = np.square(y_pred - y).sum()

    print(t, loss)



    # Backprop to compute gradients of w1 and w2 with respect to loss

    grad_y_pred = 2.0 * (y_pred - y)

    grad_w2 = h_relu.T.dot(grad_y_pred)

    grad_h_relu = grad_y_pred.dot(w2.T)

    grad_h = grad_h_relu.copy()

    grad_h[h < 0] = 0

    grad_w1 = x.T.dot(grad_h)



    # Update weights

    w1 -= learning_rate * grad_w1

    w2 -= learning_rate * grad_w2

    

end_time=time.time()

print(end_time-start_time)
import torch

import time



dtype = torch.float

device = torch.device("cpu")



# N is batch size; D_in is input dimension;

# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 64, 1000, 100, 10



# Create random input and output data

x = torch.randn(N, D_in, device=device, dtype=dtype)

y = torch.randn(N, D_out, device=device, dtype=dtype)



# Randomly initialize weights

w1 = torch.randn(D_in, H, device=device, dtype=dtype)

w2 = torch.randn(H, D_out, device=device, dtype=dtype)



start_time=time.time()

learning_rate = 1e-6

for t in range(1):

    # Forward pass: compute predicted y

    h = x.mm(w1)

    h_relu = h.clamp(min=0)

    y_pred = h_relu.mm(w2)



    # Compute and print loss

    loss = (y_pred - y).pow(2).sum().item()

    print(t, loss)



    # Backprop to compute gradients of w1 and w2 with respect to loss

    grad_y_pred = 2.0 * (y_pred - y)

    grad_w2 = h_relu.t().mm(grad_y_pred)

    grad_h_relu = grad_y_pred.mm(w2.t())

    grad_h = grad_h_relu.clone()

    grad_h[h < 0] = 0

    grad_w1 = x.t().mm(grad_h)



    # Update weights using gradient descent

    w1 -= learning_rate * grad_w1

    w2 -= learning_rate * grad_w2

    

end_time=time.time()

print(end_time-start_time)
import torch

import time



dtype = torch.float

device = torch.device("cuda:0")



# N is batch size; D_in is input dimension;

# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 64, 1000, 100, 10



# Create random input and output data

x = torch.randn(N, D_in, device=device, dtype=dtype)

y = torch.randn(N, D_out, device=device, dtype=dtype)



# Randomly initialize weights

w1 = torch.randn(D_in, H, device=device, dtype=dtype)

w2 = torch.randn(H, D_out, device=device, dtype=dtype)



start_time=time.time()

learning_rate = 1e-6

for t in range(1):

    # Forward pass: compute predicted y

    h = x.mm(w1)

    h_relu = h.clamp(min=0)

    y_pred = h_relu.mm(w2)



    # Compute and print loss

    loss = (y_pred - y).pow(2).sum().item()

    print(t, loss)



    # Backprop to compute gradients of w1 and w2 with respect to loss

    grad_y_pred = 2.0 * (y_pred - y)

    grad_w2 = h_relu.t().mm(grad_y_pred)

    grad_h_relu = grad_y_pred.mm(w2.t())

    grad_h = grad_h_relu.clone()

    grad_h[h < 0] = 0

    grad_w1 = x.t().mm(grad_h)



    # Update weights using gradient descent

    w1 -= learning_rate * grad_w1

    w2 -= learning_rate * grad_w2

    

end_time=time.time()

print(end_time-start_time)