import torch
# Number

t1 = torch.tensor(4.)

t1
# Vector

t2 = torch.tensor([1., 2, 3, 4])

t2

t21 = torch.tensor([0.1, 2, 45, 65])

t21
# Matrix

t3 = torch.tensor([[5., 6], 

                   [7, 8], 

                   [9, 10]])

print(t3.shape)

t31 = torch.tensor([[2, 4],[3.4, 0],[5, 10],[3, 4]])

t31.shape
# 3-dimensional array

t4 = torch.tensor([

    [[11, 12, 13], 

     [13, 14, 15]], 

    [[15, 16, 17], 

     [17, 18, 19.]]])

t4
# Create tensors.

x = torch.tensor(3.)

w = torch.tensor(4., requires_grad=True)

b = torch.tensor(5., requires_grad=True)

x, w, b
# Arithmetic operations

y = w * x + b

y
# Compute derivatives

y.backward()
# Display gradients

print('dy/dx:', x.grad)

print('dy/dw:', w.grad)

print('dy/db:', b.grad)
import numpy as np



x = np.array([[1, 2],[3, 4.]])

x

x1 = np.array([[4.0, 1],[5, 10]])

x1.shape

x2 = x*x1

x2
# Convert the numpy array to a torch tensor.

y2 = torch.from_numpy(x)

y2
x.dtype, y2.dtype
# Convert a torch tensor to a numpy array

z = y2.numpy()

z
z.dtype