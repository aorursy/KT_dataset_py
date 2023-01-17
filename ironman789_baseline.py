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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1)
torch.manual_seed(1)
if device == 'cuda':
  torch.cuda.manual_seed_all(1)
# 학습 파라미터 설정
learning_rate = 0.001
training_epochs = 1000
batch_size = 100
drop_prob = 0.0
train_data=pd.read_csv('../input/white-wine-quality-evalutation/train.csv',header=None,skiprows=1, usecols=range(1,13))
test_data=pd.read_csv('../input/white-wine-quality-evalutation/test.csv',header=None,skiprows=1, usecols=range(1,12))
x_train_data=train_data.loc[:,0:11]
y_train_data=train_data.loc[:,12]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(11,512,bias=True)
linear2 = torch.nn.Linear(512,512,bias=True)
linear3 = torch.nn.Linear(512,512,bias=True)
linear4 = torch.nn.Linear(512,512,bias=True)
linear5 = torch.nn.Linear(512,11,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=drop_prob)
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
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(training_epochs+1):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.view(-1, 11).to(device)
        # one-hot encoding되어 있지 않음
        Y = Y.to(device)        

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
    if epoch % 10 == 0:
      print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
# Test the model using test sets
with torch.no_grad():

  x_test_data=test_data.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float()

  prediction = model(x_test_data)
  prediction = torch.argmax(prediction, dim=1)
ans = pd.read_csv('../input/camparision/solution.csv')
ans = ans.loc[:,'quality']
ans = np.array(ans)
ans = torch.torch.from_numpy(ans).float()
print(ans)
correct_prediction = prediction.float() == ans
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.2f}% for the training set.'.format(accuracy * 100))
prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('../input/white-wine-quality-evalutation/sample_submission.csv')
submit
for i in range(len(prediction)):
  submit['quality'][i]=prediction[i].item()

submit
submit.to_csv('baseline.csv',index=False,header=True)

##!kaggle competitions submit -c white-wine-quality-evalutation -f submission.csv -m "Message"