import torch
torch.HalfTensor      # 16 бит, floating point

torch.FloatTensor     # 32 бита, floating point

torch.DoubleTensor    # 64 бита, floating point



torch.ShortTensor     # 16 бит, integer, signed

torch.IntTensor       # 32 бита, integer, signed

torch.LongTensor      # 64 бита, integer, signed



torch.CharTensor      # 8 бит, integer, signed

torch.ByteTensor      # 8 бит, integer, unsigned
a = torch.FloatTensor([1, 2])

a
a.shape
b = torch.FloatTensor([[1,2,3], [4,5,6]])

b
b.shape
x = torch.FloatTensor(2,3,4)
x
x = torch.FloatTensor(100)

x
x = torch.IntTensor(45, 57, 14, 2)

x.shape
x = torch.IntTensor(3, 2, 4)

x
x = torch.IntTensor(3, 2, 4).zero_()

x
b
b.view(3, 2)
b.view(-1)
b
a = torch.FloatTensor([1.5, 3.2, -7])
a.type_as(torch.IntTensor())
a.type_as(torch.ByteTensor())
a
a = torch.FloatTensor([[100, 20, 35], [15, 163, 534], [52, 90, 66]])

a
a[0, 0]
a[0][0]
a[0:2, 0:2]
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])
a + b
a.add(b)
b = -a

b
a + b
a - b
a.sub(b)
a * b
a.mul(b)
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])
a / b
a.div(b)
a
b
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])
a == b
a != b
a < b
a > b
a[a > b]
b[a == b]
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
a.sin()
torch.sin(a)
a.cos()
a.exp()
a.log()
b = -a

b
b.abs()
a.sum()
a.mean()
a
a.sum(dim=0)
a.sum(1)
a.max()
a.max(0)
a.min()
a.min(0)
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

a
a.t()
a
a = torch.FloatTensor([1, 2, 3, 4, 5, 6])

b = torch.FloatTensor([-1, -2, -4, -6, -8, -10])
a.dot(b)
a.shape, b.shape
a @ b
type(a)
type(b)
type(a @ b)
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

b = torch.FloatTensor([[-1, -2, -3], [-10, -20, -30], [100, 200, 300]])
a.mm(b)
a @ b
a
b
a = torch.FloatTensor([[1, 2, 3], [10, 20, 30], [100, 200, 300]])

b = torch.FloatTensor([[-1], [-10], [100]])
print(a.shape, b.shape)
a @ b
b
b.view(-1)
a @ b.view(-1)
a.mv(b.view(-1))
import numpy as np



a = np.random.rand(3, 3)

a
b = torch.from_numpy(a)

b
b -= b

b
a
a = torch.FloatTensor(2, 3, 4)

a
type(a)
x = a.numpy()

x
x.shape
type(x)
def forward_pass(X, w):

    return torch.sigmoid(X @ w)
X = torch.FloatTensor([[-5, 5], [2, 3], [1, -1]])

w = torch.FloatTensor([[-0.5], [2.5]])

result = forward_pass(X, w)

print('result: {}'.format(result))
x = torch.FloatTensor(1024, 1024).uniform_()

x
x.is_cuda
x = x.cuda()
x
a = torch.FloatTensor(10000, 10000).uniform_()

b = torch.FloatTensor(10000, 10000).uniform_()

c = a.cuda().mul(b.cuda()).cpu()
c
a
a = torch.FloatTensor(10000, 10000).uniform_().cpu()

b = torch.FloatTensor(10000, 10000).uniform_().cuda()
a + b
x = torch.FloatTensor(5, 5, 5).uniform_()



# check for CUDA availability (NVIDIA GPU)

if torch.cuda.is_available():

    # get the CUDA device name

    device = torch.device('cuda')          # CUDA-device object

    y = torch.ones_like(x, device=device)  # create a tensor on GPU

    x = x.to(device)                       # or just `.to("cuda")`

    z = x + y

    print(z)

    # you can set the type while `.to` operation

    print(z.to("cpu", torch.double))
from torch.autograd import Variable
from torch.autograd import Variable
dtype = torch.float

device = torch.device("cuda:0")

# device = torch.device("cuda:0") # Uncomment this to run on GPU



# N is batch size; D_in is input dimension;

# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 64, 3, 3, 10



# Create random Tensors to hold input and outputs.

# Setting requires_grad=False indicates that we do not need to compute gradients

# with respect to these Tensors during the backward pass.

x = torch.randn(N, D_in, device=device, dtype=dtype)

y = torch.randn(N, D_out, device=device, dtype=dtype)



# Create random Tensors for weights.

# Setting requires_grad=True indicates that we want to compute gradients with

# respect to these Tensors during the backward pass.

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)

w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)



# x = torch.FloatTensor(3, 1).uniform_()

# y = torch.FloatTensor(3, 1).uniform_()

# w = torch.FloatTensor(3, 3).uniform_() 

# b = torch.FloatTensor(3, 1).uniform_()



# x = Variable(x, requires_grad=True)

# y = Variable(x, requires_grad=False)

# w = Variable(w, requires_grad=True)

# b = Variable(b, requires_grad=True)



y_pred = (x @ w1).clamp(min=0).mm(w2)



loss = (y_pred - y).pow(2).sum()

# calculate the gradients

loss.backward()
print( Variable((y_pred - y).pow(2).sum()) )
loss.grad
w1.grad
b.grad
y.grad
loss.grad
x
x.data