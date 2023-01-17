import numpy as np

import pandas as pd

import torch

import matplotlib.pyplot as plt
# 准备数据

x_r = np.arange(-10,10,0.4)

y_r = x_r*5 + 5*np.random.rand(len(x_r))

plt.plot(x_r,y_r,'x')
# 初始化模型、优化方式、损失函数

model = torch.nn.Linear(1,1)

optimizer = torch.optim.SGD(model.parameters(),lr=6e-3)

loss_fn = loss_fn=torch.nn.MSELoss(reduction='mean')
# 将数据转为torch可用的tensor

x = torch.tensor(x_r,dtype=torch.float).reshape((-1,1))

y = torch.tensor(y_r,dtype=torch.float).reshape((-1,1))
for i in range(1000):

    # 预测

    y_pred = model(x)

    # 计算损失

    loss = loss_fn(y_pred,y)

    # 打印损失度

    if i%100 == 0:

        print('Epoch ',i,' loss is :',loss.item())

    # 梯度清0，防止与已有梯度叠加

    optimizer.zero_grad()

    # 回传梯度值，计算d(loss)/dx

    loss.backward()

    # 根据回传的梯度值，更新参数

    optimizer.step()
# tensor转numpy - ndarray

yp = y_pred.clone().reshape(-1).detach().cpu().numpy()

# 绘图

plt.figure(figsize=(10,10))

plt.plot(x_r,y_r,'x')

plt.plot(x_r,yp,'-')