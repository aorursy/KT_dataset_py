import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
# GPU를 사용하고, Python, Torch, GPU의 랜덤 시드를 사용함

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 0.01
training_epochs = 300
batch_size = 50
# 한글이 들어있어 encoding을 해주어야 함!

train_data=pd.read_csv('train.CSV',encoding='euc-kr',header=None, skiprows=1, usecols=range(0,8))
test_data=pd.read_csv('test.CSV',encoding='euc-kr',header=None, skiprows=1, usecols=range(0,7))
# train_data 전처리

train_data[2] = train_data[2]/100
train_data[4] = train_data[4]/10
train_data[6] = train_data[6]/10
train_data[7] = train_data[7]/100 * 10

# test_data 전처리

test_data[2] = test_data[2]/100
test_data[4] = test_data[4]/10
test_data[6] = test_data[6]/10
# Tensor형 데이터로 변형

x_train_data=train_data.loc[:,2:6]
y_train_data=train_data[7]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)

# TensorDataset을 이용해 Data_loader를 사용할 수 있게 만들어 줌

train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

# 데이터 로더

data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# 단일 레이어를 사용

linear1 = torch.nn.Linear(5,1,bias=True)

# Random Init => Xavier Init
torch.nn.init.xavier_uniform_(linear1.weight)

# relu는 마지막 레이어에서 뺄 것

model = torch.nn.Sequential(linear1).to(device)


loss = torch.nn.MSELoss().to(device) # MSELoss를 사용 (Mean Square)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()

        hypothesis = model(X)
        cost = loss(hypothesis, Y)
        cost.backward()
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

with torch.no_grad():

  x_test_data=test_data.loc[:,2:6]
  x_test_data=np.array(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
    
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('/content/submit_sample.CSV')

# 다시 전처리하기
for i in range(len(correct_prediction)):
  submit['total'][i]=(correct_prediction[i].item())*100 % 10


submit.to_csv('baseline.csv',index=False,header=True)

!kaggle competitions submit -c 2020-ai-term-project-18011759 -f baseline.csv -m "submit"