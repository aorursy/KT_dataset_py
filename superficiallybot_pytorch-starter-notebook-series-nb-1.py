# Import pytorch and numpy

import torch

import numpy as np
a = torch.randint(low = 0, high = 10, size = (1,))

print(a)
print(type(a))
b = torch.tensor(5)

print(b)

print(type(b))
# Adding two tensors



print(a + b)

print(torch.add(a,b))
# Subtraction



print(a - b)

print(torch.sub(a,b))
# Absolute Difference

print(torch.abs(a - b))



# or



print(torch.abs(torch.sub(a,b)))
a = torch.randn((3,3))

print(a)
b = torch.randn((3,3))

print(b)
# MATRIX OPERATIONS
# Addition

print(a + b)

print('\n', torch.add(a,b))
# Subtraction



print(a - b)

print('\n', torch.sub(a,b))
# MATRIX MULTIPLICATION



print('\n', torch.mm(a,b))
# ALSO Matrix Multiplication



print(a @ b)
# Dot product

np.dot(a,b)
# element-wise multiplication



a * b
# Transpose operation



np.transpose(a)
# Also transpose operation

torch.t(a)
# concatenation of tensors
#row-stacking, i.e., axis = 0

torch.cat((a,b))
#column-stacking, i.e., axis = 1

torch.cat((a,b), axis = 1)
# reshaping of tensors



#while reshaping ensure the new dimensions product maintains the element count

print(a.reshape(1,9))
# tensor without autograd



a = torch.rand(3,3)

print(a)
# tensor with autograd



a = torch.rand(3,3, requires_grad = True)

print(a)
# Let's begin with a simple linear equation y = x+ 5



x = torch.ones(3,3, requires_grad = True)

y = x + 5

print(y)
print(y.grad_fn)
print(y.requires_grad)
# Gradients and Backpropagation



x = torch.ones(2,2, requires_grad = True)

y = x + 3

z = y**2



res = z.mean()

print(z)

print(res)
# backpropagate and print the gradients

res.backward()

print(x.grad)
torch.zeros([2,4],dtype = torch.int32)
torch.ones([2,4], dtype = torch.float64)
x = torch.tensor([[1,2,3], [4,5,6]])

print(x[1][2])
# Numpy to Torch format

array = np.arange(1,11)

tensor = torch.from_numpy(array)

print(tensor)
# Torch to numpy



print(torch.Tensor.cpu(tensor).detach().numpy())

print(type(torch.Tensor.cpu(tensor).detach().numpy()))