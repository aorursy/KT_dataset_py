from mxnet import nd
nd.array(((1, 2, 3), (4, 5, 6)))
x = nd.ones((2, 3))
x
y = nd.random.uniform(-1, 1, (2, 3))
y
x = nd.full((2, 3), 2.0)
x
(x.shape, x.size, x.dtype)
x * y
y.exp()
nd.dot(x, y.T)
y[1, 2]
y[:, 1:3]
y[:, 1:3] = 2
y
y[1:2, 0:2] = 4
y
a = x.asnumpy()
(type(a), a)
nd.array(a)
from mxnet import nd
from mxnet.gluon import nn
layer = nn.Dense(2)
layer
layer.initialize()
x = nd.random.uniform(-1, 1, (3, 4))
layer(x)
layer.weight.data()
net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation='relu'),
    nn.Dense(84, activation='relu'),
    nn.Dense(10)
)
net
net.initialize()
x = nd.random.uniform(shape=(4, 1, 28, 28))
y = net(x)
y.shape
net[0].weight.data().shape, net[5].bias.data().shape
class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(
            nn.Dense(3, activation='relu'),
            nn.Dense(4, activation='relu')
        )
        self.dense = nn.Dense(5)
        
    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)
    
net = MixMLP()
net
net.initialize()
x = nd.random.uniform(shape=(2, 2))
net(x)
net.blk[1].weight.data()
from mxnet import nd
from mxnet import autograd
x = nd.array([[1, 2], [3, 4]])
x
x.attach_grad()
with autograd.record():
    y = 2 * x * x
y.backward()
x.grad
def f(a):
    b = a * 2
    while(b.norm().asscalar() < 1000):
        b = b * 2
    if b.sum().asscalar() >= 0:
        c = b[0]
    else:
        c = b[1]
    return c
a = nd.random.uniform(shape=2)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
[a.grad, c / a]