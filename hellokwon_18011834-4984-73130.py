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
!kaggle competitions download -c sejongyjelectricpowerprediction
!unzip sejongyjelectricpowerprediction.zip
import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 0.5
training_epochs = 50
batch_size =10
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('electric_power_train_data.csv')
#test_data=pd.read_csv('electric_power_test_data.csv')
test_data=pd.read_csv('electric_power_test_data.csv')
train_data['Date'] = train_data['Date']%1000000/10000
test_data['Date'] = test_data['Date']%1000000/10000

x_train_data=train_data.loc[:,[i for i in train_data.keys()[:-1]]]
y_train_data=train_data[train_data.keys()[-1]]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)  ####baseline??? ????????? ????????? Scaler ????????????

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(3,3,bias=True)
linear2 = torch.nn.Linear(3,1,bias=True)
relu= torch.nn.ReLU()

torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
model = torch.nn.Sequential(linear1, relu,
                            linear2).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # ??????????????? ?????????
        optimizer.zero_grad()
        # Forward ??????
        hypothesis = model(X)
        # Error ??????
        cost = loss(hypothesis, Y)
        # Backparopagation
        cost.backward()
        # ????????? ??????
        optimizer.step()

        # ?????? Error ??????
        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
with torch.no_grad():

  x_test_data=test_data.loc[:,[i for i in test_data.keys()[:]]]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('electric_power_submit_data.csv')
for i in range(len(correct_prediction)):
  submit['ElectricPower'][i]=correct_prediction[i].item()

submit
submit.to_csv('submit.csv', index = None, header=True)
!kaggle competitions submit -c sejongyjelectricpowerprediction -f submit.csv -m "Message"
