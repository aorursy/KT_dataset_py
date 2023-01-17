# kaggle 재설치
! pip uninstall kaggle -y
! pip install kaggle==1.5.6

# kaggle에서 데이터 다운로드
! mkdir ~/.kaggle
! cp drive/My\ Drive/Colab\ Notebooks/kaggle.json ~/.kaggle

! kaggle competitions download -c 2020-ai-air-pollution
! unzip 2020-ai-air-pollution
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
  y_temp = torch.FloatTensor(y_temp)
  y_train.append(y_temp)

x_train = np.array(x_train)
x_train = torch.FloatTensor(x_train)
import pandas as pd
import numpy as np

import torch
import torch.optim as optim

torch.manual_seed(777)
minw = [0, 0, 0, 0, 0, 0] # 최소의 cost를 가질 때의 W
minb = [0, 0, 0, 0, 0, 0] # 최소의 cost를 가질 때의 b
lr = [1e-7, 1e-7, 1e-7, 1e-7, 1e-5, 1e-5] # 각 수치의 learning rate
total_epochs = [1500, 1500, 1500, 1500, 1000, 1000] # 각 수치의 epoch
print_per_epoch = [i / 10 for i in total_epochs] # 각 수치의 학습 진행도를 나타내려 따로 만든 변수
# 건너 뛰어도 되는 수치의 인덱스 저장
skiptype = []

for pollution_type in range(6):
  # 0 ~ 5 : SO2, CO, O3, NO2, PM10, PM2.5

  if pollution_type in skiptype:
    continue

  minw[pollution_type] = 0
  minb[pollution_type] = 0


  W = torch.zeros((2, 1), requires_grad=True)
  b = torch.zeros(1, requires_grad=True)
  optimizer = optim.SGD([W, b], lr=lr[pollution_type])

  mincost = 1e10 # cost 최솟값

    # 시작 로그 출력
  if pollution_type == 0:   print('Start training SO2')
  elif pollution_type == 1: print('Start training CO')
  elif pollution_type == 2: print('Start training O3')
  elif pollution_type == 3: print('Start training NO2')
  elif pollution_type == 4: print('Start training PM10')
  elif pollution_type == 5: print('Start training PM2.5')

  for epoch in range(total_epochs[pollution_type] + 1):

    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train[pollution_type]) ** 2)

    # cost가 정상적인 값이고, 현재 cost가 기존 cost보다 작아지면
    # mincost, minw, minb를 갱신
    if cost != np.nan and cost < mincost:
      mincost = cost
      minw[pollution_type] = W
      minb[pollution_type] = b

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % print_per_epoch[pollution_type] == 0:
      print('Epoch {:6d}/{} , cost = {}'.format(epoch, total_epochs[pollution_type], cost.item()))
x_test = pd.read_csv('air_pollution_test.csv', header=None, skiprows=1)
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test)

predict = []
for i in range(6):
  predict.append(x_test.matmul(minw[i]) + minb[i])

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
  predict[i] = predict[i].detach().numpy().reshape(-1, 1)

id = np.array([i for i in range(len(x_test))]).reshape(-1, 1)
result = np.hstack([id, predict[0], predict[1], predict[2], predict[3], predict[4], predict[5]])

sub = pd.DataFrame(result, columns=["Id", "SO2", "CO", "O3", "NO2", "PM10", "PM2.5"])
sub['Id'] = sub['Id'].astype(int)

sub
sub.to_csv('baseline.csv', index=False)