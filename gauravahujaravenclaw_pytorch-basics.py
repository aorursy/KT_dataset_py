# Uncomment the command below if PyTorch is not installed

# !conda install pytorch cpuonly -c pytorch -y
import torch
# Number



t1 = torch.tensor(4.)

t1
t1.dtype
# Vector

t2 = torch.tensor([1., 2, 3, 4])

t2
# Matrix

t3 = torch.tensor([[5., 6], 

                   [7, 8], 

                   [9, 10]])

t3
# 3-dimensional array

t4 = torch.tensor([

    [[11, 12, 13], 

     [13, 14, 15]], 

    [[15, 16, 17], 

     [17, 18, 19.]]])

t4
print(t1)

t1.shape
print(t2)

t2.shape
print(t3)

t3.shape
print(t4)

t4.shape
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



x = np.array([[1, 2], [3, 4.]])

x
# Convert the numpy array to a torch tensor.

y = torch.from_numpy(x)

y
x.dtype, y.dtype
# Convert a torch tensor to a numpy array

z = y.numpy()

z