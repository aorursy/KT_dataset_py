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

data = pd.read_csv('/kaggle/input/metro/train.csv', encoding='utf-8')
data = data.drop(['역번호', '역이름', '호선', '계', '일평균'], axis = 1)
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import random

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

num_data = data.to_numpy()
x_data = num_data[:,:-1];
y_data = num_data[:,[-1]]

x_data = torch.FloatTensor(x_data).to(device)
y_data = torch.FloatTensor(y_data).to(device)

linear1 = torch.nn.Linear(7, 32, bias=True)
linear2 = torch.nn.Linear(32, 32, bias=True)
linear3 = torch.nn.Linear(32, 32, bias=True)
output = torch.nn.Linear(32, 1, bias=True)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(output.weight)


relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu,
                            linear2, relu,
                            linear3, relu,
                            output).to(device)

loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 15001;
for epoch in range(epochs):
  optimizer.zero_grad();
  hypothesis = model(x_data);
  cost = loss(hypothesis, y_data)
  cost.backward();
  optimizer.step()

  if(epoch%1000 == 0):
    print('Epoch: {}, Cost: {}'.format(epoch, cost.item()))
test = pd.read_csv('/kaggle/input/metro/test.csv', encoding='utf-8')
test = test.drop(['역이름', '역번호', '계', '일평균', '호선'], axis=1)

test = test.to_numpy()
test = torch.FloatTensor(test).to(device)

prediction = model(test)
submit = pd.read_csv('/kaggle/input/metro/submit.csv')

for i in range(len(submit)):
    submit['Expected'][i] = prediction[i];

submit