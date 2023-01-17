# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch
x = torch.tensor([2.,], requires_grad=True)

y = torch.tensor([1.,2.,3.], requires_grad=True)

z = torch.tensor([3.,2.,1], requires_grad=True)
def grad_example(x, y, z):

    multiply = torch.dot(y,z) #умножение y и z

    sin = torch.sin(multiply) #синус от произведения

    subtractions = torch.sub(3, sin) #вычитание, 3 - sin

    pow_ = torch.pow(subtractions,x) #возведение в степень х

    res = torch.add(2*x, pow_) #сумма

    print(res)

    res.backward()

    print("x grad =",x.grad.item())

    print("y grad =",y.grad)

    print("z grad =",z.grad)
grad_example(x, y, z)