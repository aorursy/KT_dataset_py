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
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
# 학습 파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 150
train_data=pd.read_csv('mnist_train_label.csv',header=None)
test_data=pd.read_csv('mnist_test.csv',header=None)
x_train_data=train_data.loc[:,1:785]
y_train_data=train_data.loc[:,0]

x_train_data=np.array(x_train_data,dtype=float)
y_train_data=np.array(y_train_data,dtype=float)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)
print(x_train_data)
print(y_train_data)
#단층, SGD
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(784,10,bias=True)
relu = torch.nn.ReLU()
torch.nn.init.kaiming_normal_(linear1.weight)

# ======================================
# relu는 맨 마지막 레이어에서 빼는 것이 좋다.
# ======================================
model = torch.nn.Sequential(linear1).to(device)
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.7) 
total_batch = len(data_loader)
print(total_batch)
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
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
  correct_prediction = torch.argmax(prediction, 1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submission.csv')
submit
for i in range(len(prediction)):
  submit['Category'][i]=correct_prediction[i]
  
submit.to_csv('baseline.csv',index=False)
submit
submit.to_csv('baseline.csv',index=False,header=True)

!kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f baseline.csv -m "submit"