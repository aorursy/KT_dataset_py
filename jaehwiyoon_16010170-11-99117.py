# kaggle 재설치

! pip uninstall kaggle -y
! pip install kaggle==1.5.6
! mkdir ~/.kaggle
! cp drive/My\ Drive/Colab\ Notebooks/kaggle.json ~/.kaggle

! kaggle competitions download -c 2020-ai-air-pollution
! unzip 2020-ai-air-pollution
import pandas as pd
import numpy as np

import torch
import torch.optim as optim

import random

torch.manual_seed(777)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
# 파일 열기
data = pd.read_csv('air_pollution_train.csv', header=None, skiprows=1)

# 부적합 데이터 drop
data = data.dropna()
data = np.array(data)
data = pd.DataFrame(data)

# 학습 데이터 구성
# 총 6가지의 y를 구해야 하므로 y_train을 리스트로 구현하여 각각 저장
x_train = data.loc[:, 0:1]
print(x_train)

# 시간 컬럼의 데이터를 모두 24로 나눔
# 그 값을 일수에 넣어 x_train 컬럼을 하나로 줄이고, x_train 데이터를 연속적으로 바꿈
x_train.loc[:, 1] = x_train.loc[:, 0] + x_train.loc[:, 1] / 24
x_train = x_train.loc[:, [1]]
print(x_train)

y_train = [] # so2, co, o3, no2, pm10, pm2.5
for i in range(2, 8):
  y_temp = data.loc[:, i]
  y_temp = np.array(y_temp)
  y_temp = torch.FloatTensor(y_temp).to(device)
  y_train.append(y_temp)

x_train = np.array(x_train)
x_train = torch.FloatTensor(x_train).to(device)
lr = [1e-2, 1e-2, 1e-2, 1e-2, 1, 1] # 각 수치의 learning rate
total_epochs = [1500, 1500, 1500, 1500, 1500, 1500] # 각 수치의 epoch
print_per_epoch = [i / 10 for i in total_epochs] # 각 수치의 학습 진행도를 나타내려 따로 만든 변수
class NN(torch.nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.linear1 = torch.nn.Linear(1, 2, bias = True) # 레이어 추가
    self.linear2 = torch.nn.Linear(2,16, bias = True)
    self.linear3 = torch.nn.Linear(16, 32, bias = True)
    self.linear4 = torch.nn.Linear(32, 32, bias = True)
    self.linear5 = torch.nn.Linear(32,1, bias = True)
    self.relu = torch.nn.ReLU()
    
    torch.nn.init.xavier_uniform_(self.linear1.weight)
    torch.nn.init.xavier_uniform_(self.linear2.weight)
    torch.nn.init.xavier_uniform_(self.linear3.weight)
    torch.nn.init.xavier_uniform_(self.linear4.weight)
    torch.nn.init.xavier_uniform_(self.linear5.weight)
  def forward(self,x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    out = self.relu(out)
    out = self.linear4(out)
    out = self.relu(out)
    out = self.linear5(out)
    return out
x_test = pd.read_csv('air_pollution_test.csv', header=None, skiprows=1)

x_test.loc[:, 1] = x_test.loc[:, 0] + x_test.loc[:, 1] / 24
x_test = x_test.loc[:, [1]]

x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test).to(device)
# 건너 뛰어도 되는 수치의 인덱스 저장
skiptype = []
predict = [0, 0, 0, 0, 0, 0]

for pollution_type in range(6):
  # 0 ~ 5 : SO2, CO, O3, NO2, PM10, PM2.5

  if pollution_type in skiptype:
    continue

  # 배치 사이즈 설정
  batch_size = total_epochs[pollution_type] / 5
  batch_size = int(batch_size)

  train_set = torch.utils.data.TensorDataset(x_train, y_train[pollution_type])
  data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

  model = NN().to(device)
  
  loss = torch.nn.MSELoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr[pollution_type])

  mincost = 1e10 # cost 최솟값
  minmodel = model

    # 시작 로그 출력
  if pollution_type == 0:   print('Start training SO2')
  elif pollution_type == 1: print('Start training CO')
  elif pollution_type == 2: print('Start training O3')
  elif pollution_type == 3: print('Start training NO2')
  elif pollution_type == 4: print('Start training PM10')
  elif pollution_type == 5: print('Start training PM2.5')

  for epoch in range(total_epochs[pollution_type] + 1):

    avg_cost = 0

    for X, Y in data_loader:

      X = X.to(device)
      Y = Y.to(device)

      optimizer.zero_grad()
      hypothesis = model(X)

      cost = loss(hypothesis, Y)
      avg_cost += cost / len(data_loader)

      cost.backward()
      optimizer.step()
    
    
    if avg_cost < mincost:
      mincost = avg_cost
      minmodel = model

    if epoch % print_per_epoch[pollution_type] == 0:
      print('Epoch {:6d}/{} , cost = {}'.format(epoch, total_epochs[pollution_type], cost.item()))
      
  with torch.no_grad():
    minmodel.eval()
    predict[pollution_type] = minmodel(x_test)
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
sub.to_csv('baseline_defense_1_4.csv', index=False)