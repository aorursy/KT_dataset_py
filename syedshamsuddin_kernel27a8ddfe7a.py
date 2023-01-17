# Import torch and other required modules

import torch
# for simple linear regression

batch1 = torch.rand(1, 3, 4)

batch2 = torch.rand(1, 4, 3)

inp = torch.zeros((3, 3), dtype = torch.float)

yout = torch.addbmm(inp, batch1, batch2)

yout
batch1 = torch.rand(3, 3, 4)

batch2 = torch.rand(3, 4, 5)

inp = torch.rand(3, 5)

yout = torch.addbmm(inp, batch1, batch2, beta = 2, alpha = 0.6)

yout
inp = torch.ones((2, 2), dtype = torch.int8)

batch1 = torch.rand(3, 2, 4)

batch2 = torch.rand(3, 4, 2)

yout = torch.addbmm(inp, batch1, batch2)

yout
z = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype = torch.int8)

index = torch.tensor([1], dtype = torch.long)

yout = torch.index_select(z, 1, index)

yout
z = torch.rand(3, 3, 4)

index = torch.tensor([2], dtype = torch.long)

yout = torch.index_select(z, 0, index)

z, yout
z = torch.tensor([1, 2], dtype = torch.double)

index = torch.tensor([0], dtype = torch.long)

yout = torch.index_select(z, 1, index)

yout
z = torch.randn(5, 5)

yout = torch.eig(z)

yout
z = torch.rand(5, 5)

yout = torch.eig(z, eigenvectors = True)

z, yout
z = torch.rand(5, 3)

yout = torch.eig(z)

yout
def callable(a, b):

    return a*b
z = torch.rand(4, 4)

y = torch.zeros([4, 4])

yout = z.map_(y, callable)

yout
z = torch.rand(2, 4, 2, 1)

y = torch.rand(4, 1, 1)

print('before', z)

z.map_(y, callable)

print('after', z)
z = torch.rand(5, 4, 3, 2)

y = torch.rand(5, 1, 2)

# print(z)

z.map_(y, callable)

print(z)
z = torch.rand(4, 3)

z.requires_grad
z.requires_grad_(requires_grad = True)

z.requires_grad
x = torch.tensor(3.)

w = torch.tensor(4.)

b = torch.tensor(5.)

x, w, b
y = w * x + b

y.backward()
w.requires_grad_(requires_grad = True)

# b.requires_grad_(requires_grad = True)

y = w * x

y.backward()

w.grad
z = torch.rand(2, 3)

inp = torch.rand(2, 3)

yout = z * inp

yout.backward()
z.requires_grad_()

z.requires_grad
yout = z * inp

yout.backward()
rt = torch.tensor([1., 2., 3.], requires_grad = True)

loss = rt.pow(2).sum()

loss
loss.backward()
rt.grad
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()