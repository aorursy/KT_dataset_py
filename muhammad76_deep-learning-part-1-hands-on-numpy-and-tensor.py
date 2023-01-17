import torch
#tensor is vector, matrix or n-dimnetional array
t1 = torch.tensor(4.0)
#Getting data type of tensor
t1.dtype
#Vector
t2 = torch.tensor([5., 2, 4, 5])
t2
#matrix
t3 = torch.tensor ([[2., 3],[3,4],[4,5]])
t3
#3-dimention array
t4 = torch.tensor ([[
    [2, 3, 4],
    [3, 4, 5]
], [
    [3, 4, 5],
    [5, 6, 7]
]])

t4
print(t1)
t1.shape
print(t2)
t2.shape
print(t3)
t3.shape
print(t4)
t4.shape
x = torch.tensor (3.)
w = torch.tensor (4., requires_grad = True)
b = torch.tensor (5., requires_grad = True)
y = w * x + b
y
#compute derivatives
y.backward()
#print derivatives
print ("dy/dx", x.grad)
print ("dy/dw", w.grad)
print ("dy/db", b.grad)
import numpy as np
x = np.array ([[2, 3], [4, 5]])
x
#convert numpy array to pytorch
y = torch.from_numpy(x)
y
x.dtype, y.dtype
#converting pytorch to numpy
z = y.numpy()
z