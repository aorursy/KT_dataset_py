import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

torch.manual_seed(1)
V = [1., 2. , 3.]
V_ten = torch.tensor(V)
V_ten
M = [ [1., 2., 3.], [4., 5., 6.]]
M_ten = torch.tensor(M)
M_ten
T = [ [[1., 2., 3.], [4., 5., 6.]], [ [7., 8., 9.], [10., 11., 12.]] ]
T_ten = torch.tensor(T)
T_ten
print(V_ten[0])
print(V_ten[0].item())

print(M_ten[0])
print(T_ten[0])
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
x+y
x1 = torch.randn(2, 5)
y1 = torch.randn(3, 5)
z1 = torch.cat([x1, y1]) #by default concatenates along the first axis (row / one under another)
print(z1)
x2 = torch.randn(2, 3)
y2 = torch.randn(2, 5)
z2 = torch.cat( [x2, y2], 1) #concatenation along 2nd axis - column
print(z2)
x = torch.randn(2, 3, 4)
print(x)

print(x.view(2, 12))
#OR
print(x.view(2, -1))
x = torch.tensor([1., 2., 3.], requires_grad= True)
y = torch.tensor([4., 5., 6.], requires_grad= True)
z = x+y
print(z.grad_fn)
s = z.sum()
print(s)
print(s.grad_fn)
s.backward()
print( x.grad )
x = torch.randn(2, 2)
y = torch.randn(2, 2)
print(x.requires_grad, y.requires_grad)
print(z.grad_fn)
print('\n')
x = x.requires_grad_() #change in place the value of x.requires_grad
y = y.requires_grad_()
z = x+y
print(x.requires_grad, y.requires_grad)
print(z.grad_fn)
print(z.requires_grad)
new_z = z.detach()
print(new_z.grad_fn)
print( x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print(x.requires_grad)
    print((x**2).requires_grad)
#This is the code for tutorial from the website https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html#sphx-glr-beginner-nlp-pytorch-tutorial-py
