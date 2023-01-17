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
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(1)
device = torch.device('cuda')
train = pd.read_csv('/kaggle/input/taxi/train.csv')
test = pd.read_csv('/kaggle/input/taxi/test.csv')
cab_type = {"Uber":0, "Lyft":1}

train['cab_type']= train['cab_type'].map(cab_type)
test['cab_type']= test['cab_type'].map(cab_type)
summary_mapping = {" Clear ":0," Drizzle ":1," Foggy ":2," Light Rain ":3," Mostly Cloudy ":4," Overcast ":5," Partly Cloudy ":6," Possible Drizzle ":7," Rain ":8}

train['short_summary'] = train['short_summary'].map(summary_mapping)
test['short_summary'] = test['short_summary'].map(summary_mapping)
icon_mapping = {" clear-day ":0," clear-night ":1," cloudy ":2," fog ":3," partly-cloudy-day ":4," partly-cloudy-night ":5," rain ":6}

train['icon'] = train['icon'].map(icon_mapping)
test['icon'] = test['icon'].map(icon_mapping)
train.info()
x_data = train.loc[0:,"cab_type":"precipIntensityMax"]
y_data = train.loc[0:,"price"]

x_data = np.array(x_data)
y_data = np.array(y_data)
scaler = MinMaxScaler()

x_data = scaler.fit_transform(x_data)
x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device).reshape(-1,1)

print(x_train[:5])
print(x_train.shape)
print(y_train[:5])
print(y_train.shape)
x_test = test.loc[0:,"cab_type":]
x_test = np.array(x_test)
x_test = scaler.transform(x_test)
x_test = torch.FloatTensor(x_test).to(device)

x_test[:3]
W = torch.zeros((19,1),requires_grad=True,device = device)
b = torch.zeros(1,requires_grad=True,device = device)

optimizer = optim.Adam([W, b],lr = 5*1e-3)

nb_epochs = 20000

for epoch in range(nb_epochs + 1):

  hypo = x_train.matmul(W)+b

  #cost = F.mse_loss(hypo,y_train)
  cost = torch.mean((hypo-y_train)**2)
  
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()


  if epoch % 2000 == 0:
    print('epoch {:4d}/{} Cost : {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
  
print(W)
print(b)
predict = x_test.matmul(W)+b
predict = predict.cpu() 
predict[:3]
form = pd.read_csv('/kaggle/input/taxi/submission_form.csv')
form[:5]
for i in range(len(x_test)):
  predict[i] = predict[i]
  form['price'][i] = predict[i]

form[:5]
