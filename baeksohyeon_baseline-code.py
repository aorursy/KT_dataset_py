!pip uninstall -y kaggle
!pip install --upgrade kaggle
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c childpark
!unzip childpark.zip
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re

torch.manual_seed(1)
#데이터 로드
xy = pd.read_csv('train_classi.csv', header=None)
print(xy)

xy=pd.DataFrame.dropna(xy, axis=0, how='any', thresh=None, subset=None, inplace=False)


x_data = xy.loc[1:,1:8]
y_data = xy.loc[1:, 9]

#시간대의 데이터만을 추출합니다.
date = xy.loc[1:,0]
A = date.str.extract(r'(\d+)[:]', expand=True)  # ':' 앞 숫자만 추출
print(A)
x_data["date"] = A

x_data = x_data.apply(pd.to_numeric)
y_data = y_data.apply(pd.to_numeric)

x_data = np.array(x_data)
y_data = np.array(y_data)

# MinMaxScaler 전처리
minMaxScaler = MinMaxScaler()
print(minMaxScaler.fit(x_data))
x_data = minMaxScaler.transform(x_data)

print(x_data[25:30])
print(y_data[25:30])
print(x_data.shape)
print(y_data.shape)





#학습
#선형분류를 사용
#learning rate : 0.01
#epoch : 70000

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

nb_class = 3
nb_data = len(y_data)

W = torch.zeros((9, nb_class), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr=0.01
optimizer = optim.SGD([W, b], lr=lr)

nb_epochs = 70000
for epoch in range(nb_epochs + 1):

   hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
   
   
   #y_one_hot = torch.zeros(nb_data, nb_class)
   #y_one_hot = y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
   #cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
    


   cost = F.cross_entropy((x_train.matmul(W) + b), y_train)
    


   optimizer.zero_grad()
   cost.backward()
   optimizer.step()

   if epoch % 1000 == 0:
       print('Epoch {:4d}/{} Cost: {:.6f}'.format( epoch, nb_epochs, cost.item()))

#테스트데이터 로드
test = pd.read_csv('test.csv', header=None)
test=pd.DataFrame.dropna(test, axis=0, how='any', thresh=None, subset=None, inplace=False)
print(test)
test_data = test.loc[1:,1:8]

#train과 동일하게 시간대 데이터 추출
date_t = test.loc[1:,0]

B = date_t.str.extract(r'(\d+)[:]', expand=True)
print(B)
test_data["date"] = B

test_data = test_data.apply(pd.to_numeric)

test_data=np.array(test_data)

#정규화
minmaxScaler = MinMaxScaler()
print(minmaxScaler.fit(test_data))
test_data = minmaxScaler.transform(test_data)

print(test_data[:5])
test_data=torch.FloatTensor(test_data)
print(test_data[0])



#예측


hypothesis = F.softmax(test_data.matmul(W) + b, dim=1) 
predict = torch.argmax(hypothesis, dim=1)

predict = predict.unsqueeze(1)   

print(predict[10:20])
print(lr)
print(nb_epochs)
#제출
submit = pd.read_csv('submit.csv')

for i in range(len(predict)):
  submit['result'][i] = predict[i]
submit['result']=submit['result'].astype(int)
print(submit[:10])

submit.to_csv('submit.csv', mode='w', index=False)

!kaggle competitions submit -c childpark -f submit.csv -m "0.01 70000 minmax"