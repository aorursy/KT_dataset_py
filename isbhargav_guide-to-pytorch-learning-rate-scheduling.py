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
import torch

import matplotlib.pyplot as plt


model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

lambda1 = lambda epoch: 0.65 ** epoch

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)





lrs = []



for i in range(10):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))

    scheduler.step()



plt.plot(range(10),lrs)


model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

lmbda = lambda epoch: 0.65 ** epoch

scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

lrs = []



for i in range(10):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",0.95," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(range(10),lrs)


model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

lrs = []



for i in range(10):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",0.1 if i!=0 and i%2!=0 else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(range(10),lrs)


model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)

lrs = []



for i in range(10):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",0.1 if i in [6,8,9] else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(range(10),lrs)


model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

lrs = []





for i in range(10):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",0.1," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=100)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85)

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear')

lrs = []





for i in range(100):

    optimizer.step()

    lrs.append(optimizer.param_groups[0]["lr"])

#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])

    scheduler.step()



plt.plot(lrs)

import torch

import matplotlib.pyplot as plt



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)





lrs = []



for i in range(100):

    lr_sched.step()

    lrs.append(

        optimizer.param_groups[0]["lr"]

    )



plt.plot(lrs)
import torch

import matplotlib.pyplot as plt



model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)





lrs = []



for i in range(300):

    lr_sched.step()

    lrs.append(

        optimizer.param_groups[0]["lr"]

    )



plt.plot(lrs)