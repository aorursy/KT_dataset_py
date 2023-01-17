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
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import sys

import pandas as pd

import random

from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1)
torch.manual_seed(1)
if device == 'cuda':
  torch.cuda.manual_seed_all(1)
train = pd.read_csv('/kaggle/input/defense-project/train_data.csv')
train
pd.concat((x_train_data, test))
scaler = StandardScaler()
test = pd.read_csv('/kaggle/input/defense-project/test_data.csv')

x_train_data = train.loc[:,'# mean_0_a':'fft_749_b']
y_train_data = train.loc[:,'label']

xs_data = scaler.fit_transform(pd.concat((x_train_data, test)).values)

# train
x_train = torch.FloatTensor(xs_data[:1300])
y_train = torch.LongTensor(y_train_data.values)

epochs = 10001

W = torch.zeros((2548, 3), requires_grad=True)
b = torch.zeros(3, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

for ep in range(epochs):
    hypothesis = x_train.matmul(W) + b
    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep%1000 == 0:
        print('{:4}: loss: {:2.8f}'.format(ep, cost.item()))
test = pd.read_csv('/kaggle/input/defense-project/test_data.csv')
with torch.no_grad():
    x_test = torch.FloatTensor(xs_data[1300:])
    
    hypothesis = x_test.matmul(W) + b
    
    real_test_df = pd.DataFrame([[i, r] for i, r in enumerate(torch.argmax(hypothesis, dim=1).numpy())], columns=['Id',  'Category'])
    real_test_df.to_csv('result.csv', mode='w', index=False)