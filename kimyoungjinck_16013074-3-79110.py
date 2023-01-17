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
!pip install kaggle

from google.colab import files

files.upload()
!pip uninstall -y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6

!kaggle -v
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!ls -lha kaggle.json

!chmod 600 ~/.kaggle/kaggle.json

!ls -lha kaggle.json
!kaggle competitions download -c predict-seoul-house-price
!unzip predict-seoul-house-price
import numpy as np

import torch

import torch.optim as optim

import pandas as pd

import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화

import torchvision.datasets as data

import torchvision.transforms as transforms

import random

from torch.utils.data import  TensorDataset, DataLoader

import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정

torch.manual_seed(1)  # seed 1로 낮춤

random.seed(1)

torch.cuda.manual_seed_all(1)



learning_rate = 0.001

training_epochs = 500

batch_size = 100  # batch 사이즈 100로 바꿈

drop_prob = 0.3
xy_train = pd.read_csv('train_data.csv', header = None, skiprows=1, usecols=range(2, 8))  # year/month 와 date 데이터도 사용 / Pandas방식사용

x_data = xy_train.loc[ : , 2:6] 

y_data = xy_train.loc[ : , [7]]

x_data = np.array(x_data)

y_data = np.array(y_data)



scaler = MinMaxScaler()   #정규화

x_data = scaler.fit_transform(x_data)



x_train = torch.FloatTensor(x_data).to(device)  #FloatTensor형식으로 변환

y_train = torch.FloatTensor(y_data).to(device) 
train_dataset = TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,

                                           batch_size = batch_size, 

                                           shuffle = True, 

                                           drop_last = True)
linear1 = torch.nn.Linear(5, 32,bias=True)   # 나오는 out값 변경 2->32 그에따른 in값 32로 변경

linear2 = torch.nn.Linear(32, 32,bias=True)

linear3 = torch.nn.Linear(32, 32,bias=True)

linear4 = torch.nn.Linear(32, 32,bias=True)

linear5 = torch.nn.Linear(32, 1,bias=True)

relu = torch.nn.LeakyReLU() # leakyReLU사용



torch.nn.init.kaiming_normal_(linear1.weight)  # kaiming_normal_으로 변경

torch.nn.init.kaiming_normal_(linear2.weight)

torch.nn.init.kaiming_normal_(linear3.weight)

torch.nn.init.kaiming_normal_(linear4.weight)

torch.nn.init.kaiming_normal_(linear5.weight)



model = torch.nn.Sequential(linear1,relu,    

                            linear2,relu,

                            linear3,relu,

                            linear4,relu,

                            linear5).to(device)
loss = torch.nn.MSELoss().to(device)  #MSELoss방식 사용

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



#그래프로 나타내기 위한 변수

losses = []

model_history = []  

err_history = []



total_batch = len(data_loader)  # 데이터로더의 크기만큼 total_batch란 변수에 넣어줌



for epoch in range(training_epochs + 1):

  avg_cost = 0



  for X, Y in data_loader:

    X = X.to(device)  #gpu사용

    Y = Y.to(device)



    optimizer.zero_grad()

    hypothesis = model(X)

    cost = loss(hypothesis, Y)   # hypothesis와 y비교

    cost.backward()

    optimizer.step()



    avg_cost += cost / total_batch  #평균 코스트  값에 cost/total_batch 계속 더해줌

    

  model_history.append(model) #model_history리스트에 cost.item()값 계속 추가

  err_history.append(avg_cost) #err_history리스트에 cost.item()값 계속 추가

  

  if epoch % 100 == 0:  

    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))

  losses.append(cost.item())  #losses리스트에 cost.item()값 계속 추가

print('Learning finished')
plt.plot(losses)

plt.plot(err_history)

plt.show()  #그려서 보여줌
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test_data.csv', header = None, skiprows=1, usecols = range(2, 7))

x_data = xy_test.loc[:, 2:6]

x_data = np.array(x_data)

x_data = scaler.transform(x_data)

x_test = torch.FloatTensor(x_data).to(device)



with torch.no_grad():

    model.eval()  # 주의사항 (dropout=False)

    

    predict = best_model(x_test)
submit = pd.read_csv('submit_form.csv')

submit['price'] = submit['price'].astype(float)

for i in range(len(predict)):   # 뽑아낸 데이터 submit에 price란에다가 예측한 값을 넣어줌

  submit['price'][i] = predict[i]

submit.to_csv('submit.csv', mode = 'w', index = False, header = True)   #submit.csv파일로 예측했던값 넣어줬던 데이터를 만듬
!kaggle competitions submit -c predict-seoul-house-price -f submit.csv -m "16013074 김영진" #제출