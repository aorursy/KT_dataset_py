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
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c predict-seoul-house-price
!unzip predict-seoul-house-price.zip
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화
from sklearn import preprocessing
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from torch.utils.data import  TensorDataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 500
batch_size = 42
drop_prob = 0.3
xy_train = pd.read_csv('train_data.csv', header = None, skiprows=1, usecols=range(4, 8))
x_data = xy_train.loc[ : , 4:6]
y_data = xy_train.loc[ : , [7]]
x_data = np.array(x_data)
y_data = np.array(y_data)

scaler = preprocessing.StandardScaler() # standard 정규화 사용
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device) 
train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)
linear1 = torch.nn.Linear(3, 2,bias=True)
linear2 = torch.nn.Linear(2, 2,bias=True)
linear3 = torch.nn.Linear(2, 2,bias=True)
linear4 = torch.nn.Linear(2, 2,bias=True)
linear5 = torch.nn.Linear(2, 1,bias=True)
relu = torch.nn.ReLU()

torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
torch.nn.init.kaiming_normal_(linear5.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4,relu,
                            linear5).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []
model_history = []
err_history = []

total_batch = len(data_loader)

for epoch in range(training_epochs + 1):
  avg_cost = 0

  for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
    
  model_history.append(model)
  err_history.append(avg_cost)
  
  if epoch % 10 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
plt.plot(losses)
plt.plot(err_history)
plt.show()
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test_data.csv', header = None, skiprows=1, usecols = range(4, 7))
x_data = xy_test.loc[:, 4:6]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()  # 주의사항 (dropout=False)
    
    predict = best_model(x_test)
predict
submit = pd.read_csv('submit_form.csv')
submit['price'] = submit['price'].astype(float)
for i in range(len(predict)):
  submit['price'][i] = predict[i]
submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
!kaggle competitions submit -c predict-seoul-house-price -f submit.csv -m "18011876 이병찬"