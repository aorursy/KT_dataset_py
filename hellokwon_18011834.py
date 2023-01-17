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
!kaggle competitions download -c 2020-ai-exam-fashionmnist-1
!unzip 2020-ai-exam-fashionmnist-1.zip
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
learning_rate = 0.001
training_epochs = 15
batch_size = 100
train = pd.read_csv("mnist_train_label.csv", header=None)
test = pd.read_csv("mnist_test.csv", header=None)

print(train)

x_data = train.loc[:,1:785]
y_data = train.loc[:,0]

x_train=np.array(x_data)
y_train=np.array(y_data)

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
y_train
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear = torch.nn.Linear(784,10,bias=True)
torch.nn.init.xavier_normal_(linear.weight)
model = torch.nn.Sequential(linear).to(device)
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
with torch.no_grad():
  model.eval() 
  x_test=test.loc[:,:]
  x_test=np.array(x_test)
  x_test=torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
  correct_prediction = torch.argmax(prediction, 1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submission.csv')
submit
for i in range(len(correct_prediction)):
  submit['Category'][i]=correct_prediction[i].item()

submit
submit.to_csv('submit.csv',index=False,header=True)
! kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submit.csv -m "Message"
