import os

import numpy as np

import pandas as pd

import fastai.vision as fa

from fastai.metrics import error_rate

import matplotlib.pyplot as plt

from matplotlib import animation,rc
n=100
# 初始化

x=fa.torch.ones(n,2)

x[:,0].uniform_(-1,1)

x[:5]
a=fa.tensor(3.,2.)   # 这里的tensor必须要是float型

y=x@a+fa.torch.rand(n)  # 我们产生y的数据，并随机增加一些噪音
# 画图

plt.scatter(x[:,0],y)
# 定义loss function

def mse(y,y_hat):

    return ((y_hat-y)**2).mean()
# 随机初始化一个a

a=fa.tensor(-1.,1.)

y_hat=x@a
# 计算损失

mse(y,y_hat)
# 画图

plt.scatter(x[:,0],y)

plt.scatter(x[:,0],y_hat)
# 随机梯度下降

a=fa.nn.Parameter(a)

a  # 将a赋值给模型参数，使得我们可以做a的梯度计算
def update():

    y_hat=x@a

    loss=mse(y,y_hat)

    if t % 10==0: print(loss)

    loss.backward()

    with fa.torch.no_grad():

        a.sub_(lr*a.grad)

        a.grad.zero_()
# 进行梯度下降

from tqdm import tqdm

lr=1e-1

for t in tqdm(range(100)):

    update()
# 画图

plt.scatter(x[:,0],y)

plt.scatter(x[:,0],x@a)
# 绘制动图

a=fa.nn.Parameter(fa.tensor(-1.,1))



fig=plt.figure()

plt.scatter(x[:,0],y,c="orange")

line,=plt.plot(x[:,0],x@a)

plt.close()



def animate(i):

    update()

    line.set_ydata(x@a)

    return line,

animation.FuncAnimation(fig,animate,np.arange(0,100),interval=20)