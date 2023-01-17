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
import torch.optim as optim

xy_train=pd.read_csv('../input/prediction-of-housing-price/data_for_train.csv',header=None,skiprows=1,dtype=np.float32)
print(xy_train)
x_data=xy_train.loc[:,1:6]
y_data=xy_train[7]

print(x_data)
print(y_data)
x_data=np.array(x_data)
y_data=np.array(y_data)
x_train=torch.FloatTensor(x_data)
y_train=torch.FloatTensor(y_data)
W = torch.zeros((6, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=10e-12)

nb_epochs = 32000
for epoch in range(nb_epochs + 1):
    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%8000==0:
      print('Epoch {:4d}/{} hypothesis: {} Cost: {:.1f}'.format(
          epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
      ))
def f(x):
    return float(x)
x_test=pd.read_csv('../input/prediction-of-housing-price/test_data.csv',header=None,dtype=np.float32)
print(x_test)
x_test=x_test.loc[:,1:6]
print(x_test)
x_test=np.array(x_test)
x_train1=torch.FloatTensor(x_test)
hypothesis = x_train1.matmul(W) + b 
print(hypothesis)
result=pd.read_csv('../input/answer/test-result data.csv',sep=',',skiprows=1,header=None,dtype='float32')
y_test=result[1]
sum=0
for i in range(0,5):
  error=(y_test[i]-hypothesis.data.numpy()[i])/y_test[i]
  if error<0:
    error=error*-1
  sum+=error
print(sum/5*100)
test=pd.read_csv('../input/prediction-of-housing-price/test_sample.csv')
for i in range(len(hypothesis)):
  test['ID'][i]=i
  test['Expected'][i]=hypothesis[i].item()
test.to_csv('Submission.csv',mode='w',index=False)
test