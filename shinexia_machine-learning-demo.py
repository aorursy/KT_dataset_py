# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 训练数据

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

y = np.matmul(X, np.array([1, 2])) + 3

print(y)
# 参数w，b的初始化，这里w可以初始化为0，但在深度学习中通常会用随机数来初始化

w = np.random.uniform(-1, 1, 2)

b = np.zeros(1)

w, b
# 计算预测值

y_pred = np.dot(X, w) + b

print(y_pred)
# 通常成为残差

y_residual = y_pred - y
# 计算w的梯度

w_grad = np.dot(X.T, y_residual)

w_grad
# 计算b的梯度

b_grad = np.sum(y_residual)

b_grad
# 设置学习率

lr = 0.01
# 更新参数

w = w - lr * w_grad

b = b - lr * b_grad

w, b
# 反复迭代

lr = 0.05

for epoch in range(1200):

    y_pred = np.matmul(X, w) + b

    y_residual = y_pred - y

    w_grad = np.matmul(X.T, y_residual)

    b_grad = np.sum(y_residual)

    w = w - lr * w_grad

    b = b - lr * b_grad

    print(w, b)
# 介绍pytorch的自动梯度

import torch
# 将np.ndarray 转换为 tensor 

X_tensor = torch.from_numpy(X).float()

y_tensor = torch.from_numpy(y).float()
# 初始化参数

W = torch.randn(2, requires_grad=True)

b = torch.randn(1, requires_grad=True)

W, b
lr = 0.05

for epoch in range(1200):

    # 前向计算

    y_pred = torch.matmul(X_tensor, W) + b

    

    # 损失函数

    loss = ((y_pred - y_tensor) ** 2).mean()



    # 计算梯度，反向传播算法，一种减少计算梯度时的计算量的优化算法

    loss.backward()

    

    # 更新参数

    with torch.no_grad():

        W -= lr * W.grad

        b -= lr * b.grad

        

    # AttributeError: 'NoneType' object has no attribute 'zero_'

    # W = W - lr * W.grad

    # b = b - lr * b.grad



    # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.

    # W -= lr * W.grad

    # b -= lr * b.grad

    

    # 梯度置零

    W.grad.zero_()

    b.grad.zero_()



    print('epoch {}, loss {}, w: {}, b: {}'.format(epoch, loss.item(), W.detach().numpy(), b.detach().numpy()))
# 反向传播算法梯度计算演示
W1 = torch.randn(2, 4, requires_grad=True)

b1 = torch.randn(4, requires_grad=True)

W1, b1
W2 = torch.randn(4, 8, requires_grad=True)

b2 = torch.randn(8, requires_grad=True)

W2, b2
W3 = torch.randn(8, requires_grad=True)

b3 = torch.randn(1, requires_grad=True)

W3, b3
h1 = torch.matmul(X_tensor, W1) + b1

h1
h2 = torch.matmul(h1, W2) + b2

h2
h3 = torch.matmul(h2, W3)+ b3

h3
loss = ((h3 - y_tensor) ** 2).sum() / 2

loss
W1.register_hook(lambda grad : print('W1_grad', grad))

W2.register_hook(lambda grad : print('W2_grad', grad))

W3.register_hook(lambda grad : print('W3_grad', grad))

b1.register_hook(lambda grad : print('b1_grad', grad))

b2.register_hook(lambda grad : print('b2_grad', grad))

b3.register_hook(lambda grad : print('b3_grad', grad))
loss.backward()
res = h3 - y_tensor

res
b3_grad = res.sum()

b3_grad
W3_grad = torch.matmul(h2.T, res)

W3_grad
t2 = torch.matmul(res.view(-1, 1), W3.view(1, -1))

t2
W2_grad = torch.matmul(h1.T, t2)

W2_grad
b2_grad = t2.sum(axis=0)

b2_grad
t1 = torch.matmul(t2, W2.T)

t1
W1_grad = torch.matmul(X_tensor.T, t1)

W1_grad
b1_grad = t1.sum(axis=0)

b1_grad