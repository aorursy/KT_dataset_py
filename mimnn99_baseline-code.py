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
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(1)
# train = pd.read_csv('Solar_TrainData_3.csv', header=None, skiprows=1, usecols=range(0,9))
train = pd.read_csv('/kaggle/input/solarenergy-meteorologicalphenomenon2/Solar_TrainData_3.csv', header=None, skiprows=1, usecols=range(0,9)) # 캐글 커널에서 여느라..
train = train.dropna()
train
test = pd.read_csv('/kaggle/input/solarenergy-meteorologicalphenomenon2/Solar_TestData_2.csv', header = None, skiprows=1,usecols=range(0,8))

test
Trx = train.loc[:,1:7]
Try = train.loc[:,8:8]

Trx = np.array(Trx)
Try = np.array(Try)

train_x = torch.FloatTensor(Trx)
train_y = torch.FloatTensor(Try)
tx = test.loc[:,1:7]
tx = np.array(tx)


test_x = torch.FloatTensor(tx)
#feature, class
w = torch.zeros((7,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([w,b], lr = 1e-3, momentum=0.01)

running = 10000
for run in range(running+1):
  hypothesis = train_x.matmul(w)+b
  cost = torch.mean((hypothesis - train_y)**2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  
  if run%1000 == 0:
    print(cost)
predict = test_x.matmul(w)+b
submit = pd.read_csv('/kaggle/input/solarenergy-meteorologicalphenomenon2/Solar_SubmitForm_2.csv')
submit
MAKE = pd.read_csv('/kaggle/input/solarenergy-meteorologicalphenomenon2/Solar_TestData_2.csv', header = None, skiprows= 1) 
MAKE
for i in range(len(predict)):
  submit['Predict'][i] = predict[i].item()

submit['YYYY/MM/DD'] = MAKE[0]
submit