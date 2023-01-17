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
!kaggle competitions download -c 2020-ai-exam-fashionmnist-2
!unzip 2020-ai-exam-fashionmnist-2.zip
import torch

import torchvision.datasets as dsets

import torchvision.transforms as transforms



torch.manual_seed(1)

torch.cuda.manual_seed_all(1)



# kaggle 노트북에서는 cpu로 설정

# device = 'cuda'

device = 'cpu'
# 데이터 불러오기 및 확인하기

train_data=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/mnist_train_label.csv',header=None,skiprows=0)

test_data=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/mnist_test.csv',header=None,skiprows=0)



train_data.head()
# 맨 앞에 있는 레이블 값 분리하기

train = train_data.loc[:, 1:784]

train_label = train_data.loc[:, 0]

train_label.head()
# numpy로 변환하기

train_mnist = train.to_numpy()

train_label = train_label.to_numpy()



# 정규화 하기

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

train_mnist = scaler.fit_transform(train_mnist)



# 텐서형으로 변환하기

train_x = torch.FloatTensor(train_mnist).to(device)

train_y = torch.LongTensor(train_label).to(device)
# 784 크기를 입력으로 받고 클래스 개수인 10만큼 출력값 내보내기

linear1 = torch.nn.Linear(784, 10, bias=True)

relu = torch.nn.Sigmoid()



model = torch.nn.Sequential(linear1).to(device)
# learning_rate와 에포크 설정하기

learning_rate = 0.05

training_epoch = 15



# loss는 CrossEntropy를 사용하고 optimizer는 SGD를 사용한다

loss = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for epoch in range(training_epoch):

  optimizer.zero_grad()

  hypothesis = model(train_x)

  cost = loss(hypothesis, train_y)

  cost.backward()

  optimizer.step()



  if epoch%5 == 0:

    print('Epoch: {}, Cost: {}'.format(epoch+1, cost.item()))
test_data = test_data.to_numpy()

test_tensor = torch.FloatTensor(test_data).to(device)
# 값을 예측하고 답 얻어내기

prediction = model(test_tensor)

correct = torch.argmax(prediction, 1)



correct
# 제출 형식에 답 입력하기

submit = pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/submission.csv')



for i in range(len(submit)):

  submit['Category'][i] = correct[i].item()

submit.to_csv('submit.csv', index=False)
!kaggle competitions submit -c 2020-ai-exam-fashionmnist-2 -f submit.csv -m "Message"