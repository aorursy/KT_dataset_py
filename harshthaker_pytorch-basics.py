import torch
x = torch.empty(5, 3)
print(x)
y = torch.rand(5, 3)

y
z = x * y

z
img = torch.rand(5, 5, 3)

img
zero_tensor = torch.zeros(3, 3, dtype=torch.long)

zero_tensor
ones = torch.ones(3,3, dtype = torch.long)

ones
ones = ones.to("cuda")
print(ones)
if torch.cuda.is_available():

    zero_tensor = zero_tensor.to("cuda")
print(zero_tensor)
zero_tensor = zero_tensor.to("cpu")
print(zero_tensor)
A = torch.rand(1024, 1024, 3)
B = torch.rand(1024, 1024, 3)
import time
%%time

C = A * B
A = A.to("cuda")

B = B.to("cuda")
%%time

C = A * B
Z = torch.ones_like(A)
Z = Z.to("cuda")
%%time

Z = A * B
Z.shape
T = torch.ones_like(Z)
print(T)
P = torch.randn_like(T, dtype=torch.float)
print(P[0][0])
print(P.size())
P.add_(T) # in-place addition
x = torch.randn(4,4)
x
y = x.view(16,1)

y
y.size()
z = x.view(8,2)

z
g = x.view(-1, 8)

g
a = torch.randn(2,2)
print(a[0][0])

print(a[0][0].item()) #accessing the item from the tensor - works for scalar only
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2

print(y)
z = y * 3
o = z.mean()
print(z)

print(o)
o.backward()
print(x.grad)
print(y.grad)

print(z.grad)
x = torch.ones(1, 1, requires_grad=True)
y = x * 3

print(y)
z = y + 2

print(z)
z.backward()
print(x.grad)