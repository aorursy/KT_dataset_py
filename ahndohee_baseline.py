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
import torchvision.transforms as trasforms
import random
from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
#학습 파라미터 설정
learning_rate = 0.00003
training_epochs = 200
batch_size = 20
drop_prob = 0.3
Scaler = preprocessing.StandardScaler()
train_data = pd.read_csv('/kaggle/input/ai-project-foodpoisoning/train_AI_project.csv').dropna() #NaN값 삭제
test_data = pd.read_csv('/kaggle/input/ai-project-foodpoisoning/test_AI_porject.csv')
train_data['Year']=train_data['Year']%10000/100 #Year값 월,일 사용할수 있도록 수정하기
x_train_data = train_data.loc[:,[i for i in train_data.keys()[:-1]]]
y_train_data=train_data[train_data.keys()[-1]]
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data = torch.FloatTensor(x_train_data)
y_train_data = torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(7,7,bias=True)
linear2 = torch.nn.Linear(7,7,bias=True)
linear3 = torch.nn.Linear(7,7,bias=True)
linear4 = torch.nn.Linear(7,7,bias=True)
linear5 = torch.nn.Linear(7,7,bias=True)
linear6 = torch.nn.Linear(7,7,bias=True)
linear7 = torch.nn.Linear(7,7,bias=True)
linear8 = torch.nn.Linear(7,7,bias=True)
linear9 = torch.nn.Linear(7,7,bias=True)
linear10 = torch.nn.Linear(7,1,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=drop_prob)
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)
torch.nn.init.xavier_normal_(linear6.weight)
torch.nn.init.xavier_normal_(linear7.weight)
torch.nn.init.xavier_normal_(linear8.weight)
torch.nn.init.xavier_normal_(linear9.weight)
torch.nn.init.xavier_normal_(linear10.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4,relu,
                            linear5,relu,
                            linear6,relu,
                            linear7,relu,
                            linear8,relu,
                            linear9,relu,
                            linear10).to(device)
loss = torch.nn.MSELoss().to(device)# 손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 최적화 함수 Adam
total_batch= len(data_loader)
model.train()
for epoch in range(training_epochs):
  avg_cost = 0

  for X,Y in data_loader:
    X=X.to(device)
    Y= Y.to(device)
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/ total_batch #평균 에러
  print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning finised')
#모델 테스트
with torch.no_grad():
  test_data['Year']=test_data['Year']%10000/100 #Year 월,일값 사용하기 위해 수정하기
  x_test_data = test_data.loc[:,[i for i in test_data.keys()[:]]]
  x_test_data=np.array(x_test_data)
  x_test_data=Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('/kaggle/input/ai-project-foodpoisoning/submit_sample_AI_project.csv')

for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()
submit