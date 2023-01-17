import torch
# Construct a 5x3 matrix, uninitialized

x = torch.empty(5, 3)

print(x)
# Construct a randomly initialized matrix

x = torch.rand(5, 3)

print(x)
# Construct a matrix of zeros and datatype long

x = torch.zeros(5, 3, dtype=torch.long)

print(x)
# Construct a tensor directly from data

x = torch.tensor([1.1, 2.0, 3])

print(x)
# Create a tensor based on existing tensor. These methods will reuse the properties

# of input tensor. e.g. dtype, unless new values are provided by user

x = x.new_ones(5, 3, dtype=torch.double)

print(x)



x = torch.randn_like(x, dtype=torch.float)

print(x)
# Get size of a tensor

print(x.size())
# torch.Size is in fact a tuple. So it supports all tuple operations
# Addition: Syntax 1

y = torch.rand(5, 3)

print(x + y)
# Addition: Syntax 2

print(torch.add(x, y))
# Addition: Providing an output tensor as argument

result = torch.empty(5, 3)

torch.add(x, y, out=result)

print(result)
# Addition: in-place

y.add_(x)

print(y)
print(x[0:3])
print(x[0:1])
print(x[0:1,0:1])
x = torch.randn(4, 4)

y = x.view(16)

z = x.view(-1, 8)

print(f"x: {x.size()} \ny: {y.size()} \nz: {z.size()}")
x = torch.randn(1)

print(x)

print(x.item())
# Torch Tensor

a = torch.ones(5)

print(a)
b = a.numpy()

print(b)
# See how changing a (Torch Tensor) changes b (NumPy Array)

a.add_(1)

print(a)

print(b)
import numpy as np

a = np.ones(5)

print(a)
b = torch.from_numpy(a)

print(b)
# See how changing NumPy Array changes Torch Tensor

np.add(a, 1, out=a)

print(a)

print(b)
# let's run this cell only if CUDA is available

# We will use `torch.device` objects to move tensors in and out of the GPU

if torch.cuda.is_available():

    device = torch.device("cuda")           # a CUDA device object

    y = torch.ones_like(x, device=device)   # directly create a tensor on GPU

    x = x.to(device)                        # or just use `.to('cuda')`

    z = x + y

    print(z)

    print(z.to('cpu', torch.double))        # `.to` can also change dtype together
import torch
# Create a Tensor and set `requires_grad=True` to track computation with it

x = torch.ones(2, 2, requires_grad=True)

print(x)
# Do a tensor operation

y = x + 2

print(y)
# y was created as a result of an operation, so it has a `grad_fn`.

print(y.grad_fn)
# Do more operations on y

z = y *  y * 3

out = z.mean()



print(z, out)
a = torch.randn(2, 2)

a = ((a * 3) / (a - 1))

print(a.requires_grad)

a.requires_grad_(True)

print(a.requires_grad)

b = (a * a).sum()

print(b.grad_fn)
out.backward()
print(x.grad)
x = torch.randn(3, requires_grad=True)

y = x * 2

while y.data.norm() < 1000:

    y = y * 2

print(y)
print(x.requires_grad)

print((x * 2).requires_grad)



with torch.no_grad():

    print((x * 2).requires_grad)