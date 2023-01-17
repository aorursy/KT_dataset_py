!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!ls -lha kaggle.json
!kaggle -v
!kaggle competitions download -c 2020-abalone-age
#라이브러리 로드
!unzip 2020-abalone-age.zip
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
learning_rate = 1e-4
training_epochs = 10
batch_size = 1
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
train_data = pd.read_csv('2020-abalone-train.csv', header=None, skiprows=1)
train_data = column_sex(train_data)

x_data = np.array(train_data.loc[:,0:7]) 
y_data = np.array(train_data[[8]])

x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)


# 테스트 데이터 로드------------------------------------------
x_test = pd.read_csv('2020-abalone-test.csv', header=None)
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
linear1 =torch.nn.Linear(8,512, bias = True)
linear2 =torch.nn.Linear(512,256, bias = True)
linear3 =torch.nn.Linear(256,128, bias = True)
linear4 =torch.nn.Linear(128,128, bias = True)
linear5=torch.nn.Linear(128,1,bias = True)
relu = torch.nn.LeakyReLU()
#초기화
torch.nn.init.kaiming_uniform_(linear1.weight)
torch.nn.init.kaiming_uniform_(linear2.weight)
torch.nn.init.kaiming_uniform_(linear3.weight)
torch.nn.init.kaiming_uniform_(linear4.weight)
torch.nn.init.kaiming_uniform_(linear5.weight)
#모델 구축
model = torch.nn.Sequential(linear1, relu,
                            linear2, relu,
                            linear3, relu,
                            linear4, relu,
                            linear5).to(device)
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

prediction = model(x_test)

submit = pd.read_csv('2020-abalone-submit.csv')
for i in range(len(prediction)):
  submit['Predict'][i]=prediction[i].item()
submit.to_csv('baseline.csv',index=False,header=True)

!kaggle competitions submit -c 2020-abalone-age -f baseline.csv -m "submit"
