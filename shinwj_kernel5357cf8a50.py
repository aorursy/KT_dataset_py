!pip uninstall --y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler                 #데이터 정규화
scaler=MinMaxScaler()

xy_train=np.loadtxt('train_data.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,8))  
xy_train=np.delete(xy_train,(1),axis=1)                       #부번 제외
xy_train[:,1]=xy_train[:,1]%100                               #같은 연도니까 2019 없애기

x_train=scaler.fit_transform(xy_train[:,0:-1])                #데이터 정규화

x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(xy_train[:,[-1]])


xy_test=np.loadtxt('my_test.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,7))    #확인하면서 하기 위해 가격이 포함된 파일 사용
xy_test=np.delete(xy_test,(1),axis=1)                           #부번 제외
xy_test[:,1]=xy_test[:,1]%100                                   #같은 연도니까 2019 없애기

xy_test=scaler.fit_transform(xy_test)                           #데이터 정규화

x_test=torch.from_numpy(xy_test)
#확인  -> 크기 6

print(x_train[:3])
print(y_train[:3])
print(x_test[:3])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
W=torch.zeros((6,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

optimizer=torch.optim.SGD([W,b],lr=1e-1)

epochs=100000

for epoch in range(epochs):
  hypothesis=x_train.matmul(W)+b 
  cost=torch.mean((hypothesis-y_train)**2)

  optimizer.zero_grad()
  cost.backward()  
  optimizer.step()  

  if epoch %10000==0:
    print("Epoch : {} , Cost : {}".format(epoch,cost.item()))
print(W)
print(b)
predict=x_test.matmul(W)+b
print(predict[:5])
submit=pd.read_csv('submit_form.csv')
submit.dtypes
submit['price']=submit['price'].astype(float)
submit.dtypes
for i in range(len(predict)):
  submit['price'][i]=predict[i].item()
submit
submit.to_csv('baseline.csv',index=False,header=True)
! kaggle competitions submit -c predict-seoul-house-price -f baseline.csv -m "baseline"