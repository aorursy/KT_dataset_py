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
#라이브러리 로드
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

import numpy as np
import pandas as pd
#gpu 설정, 시드 고정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device =='cuda':
    print('cuda allowed')
    torch.cuda.manual_seed_all(777)
#학습 파라미터 설정
learning_rate = 0.01
training_epochs = 50
batch_size = 50

loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#데이터 로드 및 가공

def column_sex(xy_data):
    #<가공1> x_data의 sex데이터 ->'M':0,'F':1,'I':2로 수정
    for i in range(len(xy_data.loc[:,0])): 
        if xy_data.loc[i,0]=='M':     
            xy_data.loc[i,0] = 0      # 처음 dataFrame 로드할 때 0열은 문자열형이었으므로 저장도 문자열형으로됨->'0','1','2'
        elif xy_data.loc[i,0] =='F':
            xy_data.loc[i,0] = 1
        elif xy_data.loc[i,0] =='I':
            xy_data.loc[i,0] = 2 

    #<가공2> object형으로 인식된 칼럼 -> 숫자형으로 변경
    xy_data.loc[:,0]=pd.to_numeric(xy_data.loc[:,0])

    return xy_data


# 학습 데이터 로드------------------------------------------
train_data = pd.read_csv('../input/abalone-kernel-dataset/2020-abalone-train.csv', header=None, skiprows=1)
train_data = column_sex(train_data)

x_data = np.array(train_data.loc[:,0:7]) 
y_data = np.array(train_data[[8]])

x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)


# 테스트 데이터 로드------------------------------------------
x_test = pd.read_csv('../input/abalone-kernel-dataset/2020-abalone-test.csv', header=None)
x_test = column_sex(x_test)

x_test = np.array(x_test)
x_test = torch.from_numpy(x_test).float().to(device)

# 데이터 로더--------------------------------------------
train_dataset = torch.utils.data.TensorDataset(x_data, y_data)
data_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)

print(x_data.shape)
linear1 =torch.nn.Linear(8,4, bias = True)
linear =torch.nn.Linear(4,1,bias = True)
relu = torch.nn.ReLU()
#초기화
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear.weight)
#모델 구축
model = torch.nn.Sequential(linear1, relu,
                            linear).to(device)
#loss 함수 선택, optimizer 초기화
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# 학습
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0
    for X,Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
    
        optimizer.zero_grad()
        hypothesis = model(X)
        # cost 계산

        cost = loss(hypothesis, Y)

        # error계산
        cost.backward()
        optimizer.step()

        #평균 에러 계산
        avg_cost +=cost/total_batch

    print('Epoch {:4d}, Cost: {:.6f}'.format(epoch, cost.item()))
print('Learning end')
# submission form 로드 후 값 저장하여 제출
submit = pd.read_csv('../input/abalone-kernel-dataset/2020-abalone-submit.csv')
submit
