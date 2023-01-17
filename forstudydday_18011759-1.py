!pip uninstall --y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle

!ls -lha kaggle.json

!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download --force -c 2020-ai-exam-fashionmnist-1



!unzip 2020-ai-exam-fashionmnist-1.zip
# Library Import



import pandas as pd

import numpy as np

import torch

import torchvision.datasets as data

import torchvision.transforms as transforms

import random



torch.manual_seed(777)
# Set Learning Parameter



learning_rate = 0.01

training_epochs = 15

batch_size = 1
# Data Load



train_data=pd.read_csv('/content/mnist_train_label.csv',encoding='euc-kr',header=None, skiprows=1, usecols=range(0,10))

test_data=pd.read_csv('/content/mnist_test.csv',encoding='euc-kr',header=None, skiprows=1, usecols=range(1,10))
train_data
test_data
 # Tensor형 데이터로 변형



x_train_data=train_data.loc[:,0:8]

y_train_data=train_data[9]



x_train_data=np.array(x_train_data)

y_train_data=np.array(y_train_data)



x_train_data=torch.FloatTensor(x_train_data)

y_train_data=torch.FloatTensor(y_train_data)
x_train_data
# Neural Network 모델 정의

linear = torch.nn.Linear(9,1,bias=True)

torch.nn.init.normal_(linear.weight)

model = torch.nn.Sequential(linear).to(device) # 'cuda'

model
# 모델 학습



loss = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(),learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):

    avg_cost = 0



    for X, Y in data_loader:



        X = X.to(device)

        Y = Y.to(device)



        # 그래디언트 초기화

        optimizer.zero_grad()



        hypothesis = model(X)

        cost = loss(hypothesis, Y)

        cost.backward()

        optimizer.step()



        # 평균 Error 계산

        avg_cost += cost / total_batch



    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))



print('Learning finished')
# test_data를 모델에 넣기

with torch.no_grad():



  x_test_data=test_data.loc[:,1:10]

  x_test_data=np.array(x_test_data)

  x_test_data=torch.from_numpy(x_test_data).float()



  prediction = model(x_test_data)
x_test_data
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('/content/submission.csv')

submit
for i in range(len(correct_prediction)):

  submit['Category'][i]=correct_prediction[i].item()
submit.to_csv('submit.csv',index=False,header=True)



!kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submit.csv -m "submit"