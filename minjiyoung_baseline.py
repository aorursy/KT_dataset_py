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
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as data
import torchvision.transforms as trasforms
import random
from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
xy_train=np.loadtxt('../input/ai-project-life-environment/train data.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1,9))
x_data=torch.from_numpy(xy_train[:,0:-1])
y_data=torch.from_numpy(xy_train[:,[-1]])

xy_test=np.loadtxt('../input/ai-project-life-environment/train data.csv',delimiter=',', dtype=np.float32,skiprows=1,usecols=range(1,8))
test_x_data=torch.from_numpy(xy_test)
print(x_data)
print(y_data)
print(test_x_data)
W=torch.ones((7,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

optimizer=torch.optim.SGD([W,b], lr=1e-4, momentum=0.9)
epochs=500000
for epoch in range(epochs) :
  hypothesis =x_data.matmul(W)+b
  cost=torch.mean((hypothesis-y_data)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 50000==0:
    print("epoch:{}, cost:{}". format(epoch, cost.item()))

predict=test_x_data.matmul(W)+b
print(predict)
submit=pd.read_csv('../input/ai-project-life-environment/submit sample.csv')
for i in range(len(predict)):
  submit['Expected'][i]=predict[i].item()

submit