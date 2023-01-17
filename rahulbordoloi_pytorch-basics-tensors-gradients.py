!pip install torch
import torch
# Number
t1 = torch.tensor(4.)
t1, t1.dtype
# Vector
t2 = torch.tensor([1,2,3,4])
t2, t2.dtype
# Matrix
x = [
     [1,2,3],
     [4,5,6], 
     [7,8,9]
     ]
t3 = torch.tensor(x)
x, t3, t3.dtype
t1.shape, t2.shape, t3.shape
torch.zeros([2,4], dtype=torch.float64)
cuda0 = torch.device('cuda:0')
torch.ones([2, 4], dtype = torch.float64, device = cuda0)
# Create Tensors
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b
# Arithmetic Operations
y = w * x + b
y
# Compute Derivatives
y.backward()
# Display Gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
import numpy as np

x = np.array([[1, 2], [3, 4.]])
x
# Converting the numpy array to a torch tensor.
y = torch.from_numpy(x)        # Creates the array on the same memory allocation
y
y = torch.tensor(x)        # Creates the array on the a new memory allocation
y
x.dtype, y.dtype
# Converting a torch tensor to a numpy array
z = y.numpy()
z