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
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

xy=pd.read_csv('/kaggle/input/city-commercialchange-analysis/train.csv')
xy
corr=xy.corr(method='pearson')
corr
x_data=xy.iloc[:,0:7]    #0~7 col
y_data=xy.iloc[:,7]

x_data
y_data
x_train=np.array(x_data)
y_train=np.array(y_data)

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)

x_train[:5]
x_train.shape
y_train.shape
y_train
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
# 학습 파라미터 설정
learning_rate = 0.01
training_epochs = 55
batch_size = 100

from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()  
x_train
x_train_scaler=Scaler.fit_transform(x_train)
x_train_scaler
x_train_scaler=torch.FloatTensor(x_train_scaler)

train = torch.utils.data.TensorDataset(x_train_scaler, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#xy=train
# 3-Layer

linear1 = torch.nn.Linear(7,256,bias=True)
linear2 = torch.nn.Linear(256,256,bias=True)
linear3 = torch.nn.Linear(256,4,bias=True)
relu = torch.nn.ReLU()
# Random Init => Xavier Init
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
model = torch.nn.Sequential(linear1,relu,linear2,relu,linear3).to(device)
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

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
test=pd.read_csv('/kaggle/input/city-commercialchange-analysis/test.csv')

with torch.no_grad():
  x_test=test.loc[:,:]
  x_test=np.array(x_test)
  x_test_scaler=Scaler.transform(x_test)
  x_test_scaler=torch.from_numpy(x_test_scaler).float().to(device)

  prediction=model(x_test_scaler)
  prediction = torch.argmax(prediction, 1)

prediction
submit = pd.read_csv('/kaggle/input/city-commercialchange-analysis/submit.csv')
submit
id=np.array([i for i in range(62)]).reshape(-1,1)
prediction=prediction.reshape(-1,1)

result=np.hstack([id, prediction])
df=pd.DataFrame(result, columns=['ID','Label'])
df.to_csv('defense_submit.csv', index=False, header=True)

result
