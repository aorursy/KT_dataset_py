! mkdir -p ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json
! pip uninstall --y kaggle

! pip install --upgrade

! pip install kaggle==1.5.6
! kaggle competitions download -c 2020-ai-exam-fashionmnist-1

! unzip 2020-ai-exam-fashionmnist-1
import torch

import torch.nn as nn

import pandas as pd

import numpy as np

import random

from sklearn import preprocessing

#import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

! nvidia-smi



random.seed(777)

torch.manual_seed(777)

if device == 'cuda':

  torch.cuda.manual_seed_all(777)


# 학습 파라미터 설정

learning_rate = 0.001

training_epochs = 15

batch_size = 100
train_data=pd.read_csv('mnist_train_label.csv',header=None, usecols=range(0,785))

test_data=pd.read_csv('mnist_test.csv',header=None, usecols=range(0,784))


x_data = train_data.loc[:,1:]

y_data = train_data.loc[:,[0]]

x_data = np.array(x_data)

y_data = np.array(y_data)

x_train = torch.FloatTensor(x_data)

y_train = torch.LongTensor(y_data).squeeze()

print(x_data.shape)

print(y_data.shape)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          drop_last=True)
linear = torch.nn.Linear(784,10,bias=True) #레이어 한개

torch.nn.init.normal_(linear.weight)
model = torch.nn.Sequential(linear).to(device) # 레이어 한개 이므로
# 손실함수와 최적화 함수

# BCELoss()

loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #SGD optimizer사용


total_batch = len(data_loader)

for epoch in range(training_epochs):

    avg_cost = 0



    for X, Y in data_loader:



        # (1000, 1, 28, 28) 크기의 텐서를 (1000, 784) 크기의 텐서로 변형

        X = X.view(-1, 28 * 28).to(device)

        # one-hot encoding되어 있지 않음

        Y = Y.to(device)

        #%debug



        # 그래디언트 초기화

        optimizer.zero_grad()

        # Forward 계산

        hypothesis = model(X)

        # Error 계산

        cost = loss(hypothesis, Y)

        # Backparopagation

        cost.backward()

        # 가중치 갱신

        optimizer.step()



        # 평균 Error 계산

        avg_cost += cost / total_batch



    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))



print('Learning finished')
# Test the model using test sets

with torch.no_grad():



  x_test_data=test_data.loc[:,:]

  x_test_data=np.array(x_test_data)

  #x_test_data = Scaler.transform(x_test_data)

  x_test_data = torch.FloatTensor(x_test_data).to(device)





  #x_test_data=torch.from_numpy(x_test_data).float().to(device)



  prediction = model(x_test_data)

  correct_prediction = torch.argmax(prediction, 1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submission.csv')

submit
for i in range(len(correct_prediction)):

  submit['Category'][i] = correct_prediction[i].item()

submit.to_csv('submit.csv',index=False,header=True)
! kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submit.csv -m "."