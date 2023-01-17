import pandas as pd
train_csv = pd.read_csv("kaggle_train.csv")
test_csv = pd.read_csv("kaggle_test.csv")
count = 0
sum_rate = 0
for i in train_csv.columns:
    for j in train_csv.index:
        val = train_csv.at[j,i]
        if val != 0:
            count = count +1
            sum_rate = sum_rate + val
mean = sum_rate/count
for i in train_csv.columns:
    for j in train_csv.index:
        val = train_csv.at[j,i]
        if val == 0:
            train_csv.at[j,i] = mean

train_x = train_csv.iloc[:,:-1]
train_y = train_csv.iloc[:,[-1]]
train_x.shape
import numpy as np
np_train_x=np.array(train_x)
np_train_y=np.array(train_y)
import torch
ts_train_x=torch.FloatTensor(np_train_x)
ts_train_y=torch.FloatTensor(np_train_y)
for i in test_csv.columns:
    for j in test_csv.index:
        val = test_csv.at[j,i]
        if val == 0:
            test_csv.at[j,i] = mean
np_test_x = np.array(test_csv)
ts_test_x=torch.FloatTensor(np_test_x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 모델 초기화
W_1 = torch.zeros((9718, 1), requires_grad=True)
b_1 = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.Adam([W_1, b_1], lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = ts_train_x.matmul(W_1) + b_1

    # cost 계산
    cost = torch.mean((hypothesis - ts_train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch %100 == 0:
        # 100번마다 로그 출력
        print('Cost: {:.6f}'.format(
            cost.item()
        ))
# 모델 초기화
W_2 = torch.zeros((9718, 1), requires_grad=True)
b_2 = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.Adam([W_2, b_2], lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = ts_train_x.matmul(W_2) + b_2  + 0.3 * sum(W_2**2)

    # cost 계산
    cost = torch.mean((hypothesis - ts_train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100 == 0:
        # 100번마다 로그 출력
        print('Cost: {:.6f}'.format(
            cost.item()
        ))
# 모델 초기화
W_3 = torch.zeros((9718, 1), requires_grad=True)
b_3 = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.Adam([W_3, b_3], lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = ts_train_x.matmul(W_3) + b_3  + 0.3 * sum(abs(W_3))

    # cost 계산
    cost = torch.mean((hypothesis - ts_train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100 == 0:
        # 100번마다 로그 출력
        print('Cost: {:.6f}'.format(
            cost.item()
        ))
# 모델 초기화
W_4 = torch.zeros((9718, 1), requires_grad=True)
b_4 = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.Adam([W_4, b_4], lr=1e-5)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = ts_train_x.matmul(W_4) + b_4  + 0.3 * sum(abs(W_4))+ 0.3 * sum(W_4**2)

    # cost 계산
    cost = torch.mean((hypothesis - ts_train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100 == 0:
        # 100번마다 로그 출력
        print('Cost: {:.6f}'.format(
            cost.item()
        ))
y1 = ts_test_x.matmul(W_1) + b_1
y2 = ts_test_x.matmul(W_2) + b_2
y3 = ts_test_x.matmul(W_3) + b_3
y4 = ts_test_x.matmul(W_4) + b_4
y1_pd = pd.DataFrame(y1.detach().numpy(),
                    columns = ['result[Forrest Gump (1994)]']
                    )
y2_pd = pd.DataFrame(y2.detach().numpy(),
                    columns = ['result[Forrest Gump (1994)]']
                    )
y3_pd = pd.DataFrame(y3.detach().numpy(),
                    columns = ['result[Forrest Gump (1994)]']
                    )
y4_pd = pd.DataFrame(y4.detach().numpy(),
                    columns = ['result[Forrest Gump (1994)]']
                    )

Id = pd.DataFrame(range(0, 29),columns=['Id'])
Id['Id'].astype(int)
result_1 = pd.concat([Id,y1_pd], axis =1)
result_2 = pd.concat([Id,y2_pd], axis =1)
result_3 = pd.concat([Id,y3_pd], axis =1)
result_4 = pd.concat([Id,y4_pd], axis =1)
result_1.to_csv('Linear.csv', mode='w', index = False)
result_1.to_csv('Ridge.csv', mode='w', index = False)
result_1.to_csv('Lasso.csv', mode='w', index = False)
result_1.to_csv('Ela.csv', mode='w', index = False)
! kaggle competitions submit -c forrest-gump-1994 -f Linear.csv -m "Linear"
# ! kaggle competitions submit -c forrest-gump-1994 -f Ridge.csv -m "Ridge"
# ! kaggle competitions submit -c forrest-gump-1994 -f Lasso.csv -m "Lasso"
# ! kaggle competitions submit -c forrest-gump-1994 -f Ela.csv -m "Ela.csv"
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
training_epochs = 1000
batch_size = 100
Scaler = preprocessing.StandardScaler()
train_dataset = torch.utils.data.TensorDataset(ts_train_x, ts_train_y)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear=torch.nn.Linear(9718,1,bias=True)
dropout=torch.nn.Dropout(p=0.3)
relu=torch.nn.ReLU()
torch.nn.init.xavier_uniform_(linear.weight)

model = torch.nn.Sequential(linear,dropout).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
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
    if epoch %100 == 0:
        # 100번마다 로그 출력
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    

print('Learning finished')
np_test_x = np.array(test_csv)
with torch.no_grad():
  model.eval()
  x_test=test_csv
  x_test=np.array(x_test)
  x_test=torch.from_numpy(x_test).float().to(device)
  predict=model(x_test)
correct_prediction = predict.cpu().numpy().reshape(-1,1)
predict
y5_pd = pd.DataFrame(correct_prediction,
                    columns = ['result[Forrest Gump (1994)]']
                    )
result_5 = pd.concat([Id,y5_pd], axis =1)
result_5.to_csv('nn.csv', mode='w', index = False)
! kaggle competitions submit -c forrest-gump-1994 -f nn.csv -m "nn"
