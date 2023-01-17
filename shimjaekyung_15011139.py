
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls -lha kaggle.json

!chmod 600 ~/.kaggle/kaggle.json
!ls -lha kaggle.json
! kaggle competitions download -c childpark
!unzip childpark.zip
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
device = 'cuda'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
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


minMaxScaler = MinMaxScaler()
print(minMaxScaler.fit(x_data))
x_data = minMaxScaler.transform(x_data)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=100,
                                          shuffle=True,
                                          drop_last=True)
import torch.nn.functional as F
import torch.optim as optim

nb_class=3
nb_data=len(y_train)
l1 = torch.nn.Linear(9, 4)
l2 = torch.nn.Linear(4, nb_class)
relu = torch.nn.ReLU()
torch.nn.init.xavier_uniform_(l1.weight)
torch.nn.init.xavier_uniform_(l2.weight)
model = torch.nn.Sequential(l1, relu, l2).to(device)
model
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
total_batch = len(data_loader)
model_h = []
error_h = []
for epoch in range(1, 1+2000):
  avg_cost = 0

  for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

   
     
  optimizer.zero_grad()
  hypothesis = model(X)
        
  cost = loss(hypothesis, Y)
  cost.backward()
  optimizer.step()
  avg_cost += cost
  avg_cost /= total_batch
  model_h.append(model)
  error_h.append(avg_cost)

  if epoch % 50 == 1 :
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
best_model = model_h[np.argmin(error_h)]
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
with torch.no_grad():
    test_data = test_data.to(device)
    pred = best_model(test_data)
    predict=torch.argmax(pred,dim=1)

    print(predict.shape)
predict
submit=pd.read_csv('submit.csv')
submit
predict=predict.cpu().numpy().reshape(-1,1)

id=np.array([i for i in range(len(predict))]).reshape(-1,1)
result=np.hstack([id,predict])

submit=pd.DataFrame(result,columns=["id","result"])
submit.to_csv("submit.csv",index=False,header=True)
submit
!kaggle competitions submit -c childpark -f submit.csv -m "15011139 심재경"