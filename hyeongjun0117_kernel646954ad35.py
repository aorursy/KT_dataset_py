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

import pandas as pd

import torch.optim as optim

import numpy as np

torch.manual_seed(1)

device = torch.device("cuda")

#데이터

xy_data = pd.read_csv('../input/2020-ai-exam-cabbage/train_cabbage_price.csv')

x_test = pd.read_csv('../input/2020-ai-exam-cabbage/test_cabbage_price.csv')

submit = pd.read_csv('../input/2020-ai-exam-cabbage/submit_sample.csv')
xy_data = np.array(xy_data)

x_train = torch.FloatTensor(xy_data[:,1:-1]).to(device)

y_train = torch.FloatTensor(xy_data[:,-1]).to(device)



x_test = np.array(x_test)

x_test = torch.FloatTensor(x_test[:,1:]).to(device)
x_train.shape
W = torch.zeros((4,1)).to(device).detach().requires_grad_(True)

b = torch.zeros(1).to(device).detach().requires_grad_(True)

optimizer= optim.SGD([W,b], lr=1e-3)

nb_epochs = 1000
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

  hypothesis = x_train.matmul(W)+b

  cost = ((hypothesis-y_train)**2).mean()

  

  optimizer.zero_grad()

  cost.backward()

  optimizer.step()

  if epoch%100==0:

    print('cost = {}'.format(cost.item()))
predict = x_test.matmul(W)+b

for i in range(len(predict)):

  submit['Expected'][i]=predict[i]
submit.to_csv('submit.csv', mode='w', header= True, index= False)