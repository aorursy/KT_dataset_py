import torch
# Number

t1 = torch.tensor(4.)

t1
t1.dtype
# Vector

t2 = torch.tensor([1., 2, 3, 4])

t2
# Matrix

t3 = torch.tensor([[5., 6], [7, 8], [9, 10]])

t3
# 3-dimensional array

t4 = torch.tensor([

    [[11, 12, 13], 

     [13, 14, 15]], 

    [[15, 16, 17], 

     [17, 18, 19.]]])

t4
t1.shape
t2.shape
t3.shape
t4.shape
# Create tensors.

x = torch.tensor(3.)

w = torch.tensor(4., requires_grad=True)

b = torch.tensor(5., requires_grad=True)
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



x = np.array([[1, 2], [3, 4.]])

x
# Convert the numpy array to a torch tensor.

y = torch.from_numpy(x)

y
x.dtype, y.dtype
# Convert a torch tensor to a numpy array

z = y.numpy()

z
!pip install jovian --upgrade
import jovian
jovian.commit()