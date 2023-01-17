import torch 

import torch.nn as nn



def corr2d(X, K):

    H, W = X.shape

    h, w = K.shape

    Y = torch.zeros(H - h + 1, W - w + 1)

    for i in range(Y.shape[0]):

        for j in range(Y.shape[1]):

            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()

    return Y
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

K = torch.tensor([[0, 1], [2, 3]])

Y = corr2d(X, K)

print(Y)
class Conv2D(nn.Module):

    def __init__(self, kernel_size):

        super(Conv2D, self).__init__()

        self.weight = nn.Parameter(torch.randn(kernel_size))

        self.bias = nn.Parameter(torch.randn(1))



    def forward(self, x):

        return corr2d(x, self.weight) + self.bias
X = torch.ones(6, 8)

Y = torch.zeros(6, 7)

X[:, 2: 6] = 0

Y[:, 1] = 1

Y[:, 5] = -1

print(X)

print(Y)
conv2d = Conv2D(kernel_size=(1, 2))

step = 30

lr = 0.01

for i in range(step):

    Y_hat = conv2d(X)

    l = ((Y_hat - Y) ** 2).sum()

    l.backward()

    # 梯度下降

    conv2d.weight.data -= lr * conv2d.weight.grad

    conv2d.bias.data -= lr * conv2d.bias.grad

    

    # 梯度清零

    conv2d.weight.grad.zero_()

    conv2d.bias.grad.zero_()

    if (i + 1) % 5 == 0:

        print('Step %d, loss %.3f' % (i + 1, l.item()))

        

print(conv2d.weight.data)

print(conv2d.bias.data)
X = torch.rand(4, 2, 3, 5)

print(X.shape)



conv2d = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 5), stride=1, padding=(1, 2))

Y = conv2d(X)

print('Y.shape: ', Y.shape)

print('weight.shape: ', conv2d.weight.shape)

print('bias.shape: ', conv2d.bias.shape)
X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)

pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))

Y = pool2d(X)

print(X)

print(Y)