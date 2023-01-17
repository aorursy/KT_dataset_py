# Import torch and other required modules
import torch
# Example 1 
inp = torch.randn(2)
matrix = torch.randn(2, 3)
vector = torch.randn(3)
torch.addmv(inp, matrix, vector)
# Example 2 
inp = torch.randn(5)
matrix = torch.randn(5,5)
vector = torch.randn(5)
torch.addmv(inp, matrix, vector, beta=0.2, alpha=0.5, out = None)
# Example 3 
inp = torch.randn(4)
matrix = torch.randn(4,5)
vector = torch.randn(5)
beta = 0.2
alpha = 0.5
torch.addmv(inp, matrix, vector, beta, alpha, out = None)
# Example 1 
a = torch.arange(4.)
print('a = ', a)
print(a.dtype)
b = torch.reshape(a, (2, 2))
print('b = ', b)
# Example 2 
a = torch.randn(2,2)
print('a = ', a)
b = torch.reshape(a, (4, ))
print('b = ', b)
# Example 3 
a = torch.randn(2,2)
print('a = ', a)
b = torch.reshape(a, 4)
print('b = ', b)
# Example 1 
a = torch.randn(4)
print('a = ', a)
b = torch.rsqrt(a)
print('b = ', b)
# Example 2 
a = torch.tensor([5., 10.])
print('a = ', a)
b = torch.rsqrt(a)
print('b = ', b)
# Example 3 
a = torch.tensor([5, 10])
print('a = ', a)
b = torch.rsqrt(a)
print('b = ', b)
# Example 1 
inp = torch.tensor([2.])
print('input = ', inp)
out = torch.sigmoid(inp)
print('output = ', out)
# Example 2 
inp = torch.randn(4)
print('input = ', inp)
out = torch.sigmoid(inp)
print('output = ', out)
# Example 3 
inp = torch.randn(4.)
print('input = ', inp)
out = torch.sigmoid(inp)
print('output = ', out)
# Example 1 
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('x = ',x ,'\n')
y = torch.narrow(x, 0, 0, 2)
print('narrow dimension 0, row-wise starting from 0 to end 0+2-1')
print('y = ', y)
print('\n')
z = torch.narrow(x, 0, 1, 2)
print('narrow dimension 0, row-wise starting from 1 to end 1+2-1')
print('z =', z)
# Example 2 
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('x = ',x ,'\n')
a = torch.narrow(x, 1, 0, 2)
print('narrow dimension 1, column-wise starting from 0 to end 0+2-1')
print('a = ', a)
print('\n')
b = torch.narrow(x, 1, 1, 2)
print('narrow dimension 1, column-wise starting from 1 to end 1+2-1')
print('b = ', b)
# Example 3 
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('x = ',x ,'\n')
a = torch.narrow(x, 2, 0, 2)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
