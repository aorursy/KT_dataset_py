import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
alpha = 0.01

n_steps = 1000

target_price = 20000
x = torch.tensor([18000., 16000., 19000., 21000., 16500., 18700., 21300.])

y = torch.tensor([0., 1., 0., 0., 1., 1., 0.])
x_mean = x.mean()

x_std = x.std()



y_mean = y.mean()

y_std = y.std()
x_n = (x - x_mean)/x_std

print(x_n)
k, b = torch.tensor([1.], requires_grad=True), torch.tensor([0.], requires_grad=True)
z = k * x_n + b



y_n_ = 1/(1+torch.exp(-z))

y_n_
BCE = torch.nn.BCELoss()

loss = BCE(y_n_, y)

loss
loss = -(y * torch.log(y_n_) + (1-y) * torch.log(1-y_n_))

loss.mean()
opt = torch.optim.Adam([k, b], lr=alpha)

opt
# Ocновные расчеты тут.



alpha = 0.01

n_steps = 1000

target_price = 20000

x = torch.tensor([18000., 16000., 19000., 21000., 16500., 18700., 21300.])

y = torch.tensor([0., 1., 0., 0., 1., 1., 0.])

x_mean = x.mean()

x_std = x.std()





x_n = (x - x_mean)/x_std

k, b = torch.tensor([1.], requires_grad=True), torch.tensor([0.], requires_grad=True)

opt = torch.optim.Adam([k, b], lr=alpha)



for _ in range(n_steps):

    z = k * x_n + b

    y_n_ = 1/(1+torch.exp(-z))

#     loss = -(y * torch.log(y_n_) + (1-y) * torch.log(1-y_n_))    

    loss = BCE(y_n_, y)

    opt.zero_grad()

    loss.backward()

    opt.step()

    

    if _ % 10 == 0:

        print(_)

        print(loss)

        print('______')
k
b
z = k *  (target_price - x_mean)/x_std + b

y_n_ = 1/(1+torch.exp(-z))

y_n_


print(float(y_target), "\nДелаем вывод, что скорее всего при цене 20000 iphone за месяц не продастся" )