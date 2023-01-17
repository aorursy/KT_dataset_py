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
import torch.optim as optim

torch.manual_seed(777)
if torch.cuda.is_available() is True:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
# 파일 열기
data = pd.read_csv('air_pollution_train.csv', header=None, skiprows=1)

# 부적합 데이터 drop
data = data.dropna()
data = np.array(data)
data = pd.DataFrame(data)
data[0] = data[0].astype(int)
data[1] = data[1].astype(int)

# 학습 데이터 구성
# 총 6가지의 y를 구해야 하므로 y_train을 리스트로 구현하여 각각 저장
x_train = data.loc[:, 0:1]
y_train = [] # so2, co, o3, no2, pm10, pm2.5
for i in range(2, 8):
  y_temp = data.loc[:, i]
  y_temp = np.array(y_temp)
  y_temp = torch.FloatTensor(y_temp).to(device)
  y_train.append(y_temp)

x_train = np.array(x_train)
x_train = torch.FloatTensor(x_train).to(device)
lr = [1e-2, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1] # 각 수치의 learning rate
total_epochs = [1500, 1500, 1500, 1500, 1500, 1500] # 각 수치의 epoch
print_per_epoch = [i / 10 for i in total_epochs] # 각 수치의 학습 진행도를 나타내려 따로 만든 변수
class NN(torch.nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.linear1 = torch.nn.Linear(2,16, bias = True)
    self.linear2 = torch.nn.Linear(16,32, bias = True)
    self.linear3 = torch.nn.Linear(32,32, bias = True)
    self.linear4 = torch.nn.Linear(32,1,bias = True)
    self.relu = torch.nn.ReLU()
    
    torch.nn.init.xavier_uniform_(self.linear1.weight)
    torch.nn.init.xavier_uniform_(self.linear2.weight)
    torch.nn.init.xavier_uniform_(self.linear3.weight)
    torch.nn.init.xavier_uniform_(self.linear4.weight)
  def forward(self,x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    out = self.relu(out)
    out = self.linear4(out)
    return out

model = NN().to(device)
x_test = pd.read_csv('air_pollution_test.csv', header=None, skiprows=1)
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test).to(device)
# 건너 뛰어도 되는 수치의 인덱스 저장
skiptype = []
predict = []
count = -1
for pollution_type in range(6):
  # 0 ~ 5 : SO2, CO, O3, NO2, PM10, PM2.5

  if pollution_type in skiptype:
    continue

  loss = torch.nn.MSELoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr[pollution_type])

  mincost = 1e10 # cost 최솟값

    # 시작 로그 출력
  if pollution_type == 0:   print('Start training SO2')
  elif pollution_type == 1: print('Start training CO')
  elif pollution_type == 2: print('Start training O3')
  elif pollution_type == 3: print('Start training NO2')
  elif pollution_type == 4: print('Start training PM10')
  elif pollution_type == 5: print('Start training PM2.5')

  for epoch in range(total_epochs[pollution_type] + 1):

    hypothesis = model(x_train)
    cost = torch.sqrt((loss(hypothesis,y_train[pollution_type])))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % print_per_epoch[pollution_type] == 0:
      print('Epoch {:6d}/{} , cost = {}'.format(epoch, total_epochs[pollution_type], cost.item()))
      
  with torch.no_grad():
    model.eval()
    predict.append(model(x_test))
sub = pd.read_csv('air_pollution_submission.csv', header=None, skiprows=1)

sub[1] = sub[1].astype(float)
sub[2] = sub[2].astype(float)
sub[3] = sub[3].astype(float)
sub[4] = sub[4].astype(float)
sub[5] = sub[5].astype(float)
sub[6] = sub[6].astype(float)

sub = np.array(sub)
for i in range(len(sub)):
  sub[i][1] = predict[0][i]
  sub[i][2] = predict[1][i]
  sub[i][3] = predict[2][i]
  sub[i][4] = predict[3][i]
  sub[i][5] = predict[4][i]
  sub[i][6] = predict[5][i]

for i in range(6):
  predict[i] = predict[i].detach().cpu().numpy().reshape(-1, 1)

id = np.array([i for i in range(len(x_test))]).reshape(-1, 1)
result = np.hstack([id, predict[0], predict[1], predict[2], predict[3], predict[4], predict[5]])

sub = pd.DataFrame(result, columns=["Id", "SO2", "CO", "O3", "NO2", "PM10", "PM2.5"])
sub['Id'] = sub['Id'].astype(int)

sub
sub.to_csv('yh_submit.csv', index=False)