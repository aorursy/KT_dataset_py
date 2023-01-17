# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch



x = torch.Tensor([5,3])

y = torch.Tensor([2,1])



print(x*y)
x = torch.zeros([2,5])

print(x)
print(x.shape)
y = torch.rand([2,5])

print(y)
y.view([1,10])
y
y = y.view([1,10])

y
import torch

import torchvision

from torchvision import transforms, datasets



train = datasets.MNIST('', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))



test = datasets.MNIST('', train=False, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

                                   
for data in trainset:

    print(data)

    break
X, y = data[0][0], data[1][0]
print(data[1])
import matplotlib.pyplot as plt  # pip install matplotlib



plt.imshow(data[0][0].view(28,28))

plt.show()
data[0][0][0][0]
data[0][0][0][3]
total = 0

counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}





for data in trainset:

    Xs, ys = data

    for y in ys:

        counter_dict[int(y)] += 1

        total += 1



print(counter_dict)



for i in counter_dict:

    print(f"{i}: {counter_dict[i]/total*100.0}%")