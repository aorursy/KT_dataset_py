from __future__ import print_function

import torch
x = torch.Tensor(5, 3)

print(x)
x = torch.rand(5, 3)

print(x)
print(x.size())
y = torch.rand(5, 3)

print(x + y)
print(torch.add(x, y))
result = torch.Tensor(5, 3)

torch.add(x, y, out=result)

print(result)
y.add_(x)

print(y)
print(x[:, 1])