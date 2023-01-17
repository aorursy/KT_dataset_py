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
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

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

# 학습 파라미터 설정
learning_rate = 0.01
training_epochs = 15
batch_size = 100
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-4/mnist_train_label.csv',header=None,usecols=range(0,784))
test_data=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-4/mnist_test.csv',header=None,usecols=range(1,785))

x_train_data=train_data.loc[:,1:785]
y_train_data=train_data.loc[:,0]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)

print(x_train_data.shape)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

linear1 = torch.nn.Linear(783,512,bias=True)
linear2 = torch.nn.Linear(512,512,bias=True)
linear3 = torch.nn.Linear(512,512,bias=True)
linear4 = torch.nn.Linear(512,512,bias=True)
linear5 = torch.nn.Linear(512,10,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)

# Random Init => Xavier Init
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)

model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3,relu,dropout,
                            linear4,relu,dropout,
                            linear5).to(device)
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

total_batch = len(data_loader)
model.train() # 주의사항 drop_out = True
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        # (1000, 1, 28, 28) 크기의 텐서를 (1000, 784) 크기의 텐서로 변형
        X = X.view(-1, 783).to(device)
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
    model.eval()  # 주의사항 (dropout=False)
    x_test_data=test_data.loc[:,1:785]
    x_test_data=np.array(x_test_data)
    print(x_test_data.shape)
    x_test_data = Scaler.fit_transform(x_test_data)
    x_test_data=torch.from_numpy(x_test_data).float().to(device)

    prediction = model(x_test_data)
    correct_prediction = torch.argmax(prediction, 1)


print(correct_prediction)

submit=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-4/submission.csv')
for i in range(len(correct_prediction)):
  submit['Category'][i]=correct_prediction[i].item()
print(submit.dtypes)

submit

submit.to_csv('baseline.csv',index=False,header=True)

!kaggle competitions submit -c 2020-ai-exam-fashionmnist-4 -f submission.csv -m "Message"
