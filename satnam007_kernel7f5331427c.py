# import torch

import torch
# Number

t1=torch.tensor(5.)

t1

# t1 = torch.tensor(4.)

# t1
t1.dtype

# t1.dtype
# Vector

t2 = torch.tensor([1, 3, 4, 9.4, 4])

print(t2)

print(t2.dtype)



# t2 = torch.tensor([1., 2, 3, 4])

# t2
# Matrix

t3 = torch.tensor([[4,5],

                   [6,7],

                   [4,1],

                   [0,9],

                   [3,1]])



print(t3)

print(t3.dtype)



# t3 = torch.tensor([[5., 6], 

#                    [7, 8], 

#                    [9, 10]])

# t3
# 3-dimensional array



t4 = torch.tensor([

    [11,4,2,6],

    [0,5,5,3],

    [2,7,5,4],

    [4,5,6,23],

    [4,1,0,0]

])



print(t4)

print(t4.dtype)



# t4 = torch.tensor([

#     [[11, 12, 13], 

#      [13, 14, 15]], 

#     [[15, 16, 17], 

#      [17, 18, 19.]]])

# t4
print('t1\n',t1)

print(t1.shape,'\n')

print('t2\n',t2,'\n',t2.shape,'\n')

print('t3\n',t3,'\n',t3.shape,'\n')

print('t4\n',t4,'\n',t4.shape)
# Create tensors.

x = torch.tensor(3.)

w = torch.tensor(4., requires_grad=True)

b = torch.tensor(5., requires_grad=True)

x, w, b

y = w * x + b

y

# Compute derivatives

y.backward()

# Display gradients

print('dy/dx:', x.grad)

print('dy/dw:', w.grad)

print('dy/db:', b.grad) #3 * 4 + 5 = 17
# Create tensors.

x = torch.tensor(3.)

w = torch.tensor(4., requires_grad=True)

b = torch.tensor(5., requires_grad=True)

print(x, w, b)
# Arithmetic operations

z = w * x + b

print(z)
# Compute derivatives

z.backward()
# Display gradients

print('dy/dx:', x.grad)

print('dy/dw:', w.grad)

print('dy/db:', b.grad) #3 * 4 + 5 = 17
import numpy as np



x = np.array([

    [1,2,3],

    [3,5,6],

    [6,2,4]

 ])

x
# Convert the numpy array to a torch tensor.

y = torch.from_numpy(x)

y
x.dtype,y.dtype
# Convert a torch tensor to a numpy array

z = y.numpy()

z
!pip install jovian --upgrade --quiet

import jovian
jovian.commit(project='01-basics-pytorch')