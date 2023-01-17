from __future__ import print_function

import numpy as np

import torch
x = torch.empty(5,3)

print(x)
x = torch.zeros(5,3, dtype=torch.long)

print(x)
x = torch.tensor([[5, 3],[4,5]])

y = torch.tensor([5,3])

print("x tensor")

print(x)

print("y tensor")

print(y)
x = x.new_ones(5,3, dtype=torch.double) # new_* methods take in sizes

print(x)
x = torch.randn_like(x, dtype=torch.float) #override dtype

print(x)

print(x.size())
y = torch.rand(5, 3)

print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)

torch.add(x, y, out=result) #provide output tensor as argument

print(result)
y.add_(x)

print(y)
x = torch.rand(4,4)

y = x.view(16)

z = x.view(-1,8)

print(x.size(), y.size(), z.size())
a = torch.ones(5)

print(a)
b = a.numpy()

print(b)
a.add_(1)

print(a)

print(b)
a = np.ones(5)

b = torch.from_numpy(a)

print(a)

print(b)

np.add(a, 1, out=a)

print("after addition")

print(a)

print(b)
if torch.cuda.is_available():

    device = torch.device("cuda")

    y = torch.ones_like(x, device=device)

    x = x.to(device)

    z = x+y

    print(z)

    print(z.to("cpu", dtype=torch.double))
x = torch.ones(2,2, requires_grad=True)

print(x)
y = x + 2

print(y)

print(y.grad_fn)
z = y * y * 3

out = z.mean()

print(z, out)
a = torch.randn(2,2)

print(a.requires_grad)

a.requires_grad_(True)

print(a.requires_grad)

b = (a*a).sum()

print(b)
out.backward()

print(x.grad)