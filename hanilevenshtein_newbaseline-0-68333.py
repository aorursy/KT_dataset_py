!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

random.seed(777)
torch.manual_seed(777)

!kaggle competitions download -c aqiprediction
Scaler=preprocessing.StandardScaler()
!unzip aqiprediction.zip
train=pd.read_csv("train.csv")
print(train["PM10"].value_counts(normalize=True))

train['시간'] = train['시간'].astype(str)
train['시간'] = pd.to_datetime(train['시간'],format="%Y-%m-%d:%H", errors='ignore')
train=train.set_index('시간')

wind=pd.DataFrame({'풍향': np.cos(np.pi*train["풍향"]/360)})
train=train.drop("풍향",axis=1)
train=pd.concat((train,wind),axis=1)

train_x=train[['습도','강수','기온','풍속','풍향']]
train_y=train["PM10"].replace(["좋음","보통","나쁨","매우나쁨"], [0,1,2,3])
scaler=MinMaxScaler()
train_x=Scaler.fit_transform(train_x)
train_x=torch.FloatTensor(train_x)
train_y=torch.LongTensor(np.array(train_y))
print(train_x.shape)
train_dataset=torch.utils.data.TensorDataset(train_x,train_y)
learning_rate =0.01
training_epochs = 200
batch_size = 200
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1=torch.nn.Linear(5,4,bias=True)
linear2=torch.nn.Linear(4,4,bias=True)
linear3=torch.nn.Linear(4,4,bias=True)
linear4=torch.nn.Linear(4,4,bias=True)
linear5=torch.nn.Linear(4,4,bias=True)

relu=torch.nn.LeakyReLU()
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)
model=torch.nn.Sequential(linear1,relu,
                          linear2,relu,
                          linear3,relu,
                          linear4,relu,
                          linear5
                          )
loss = torch.nn.CrossEntropyLoss()# softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(training_epochs):
  avg_cost = 0

  for X, Y in data_loader:
      optimizer.zero_grad()
      hypothesis = model(X)
      cost = loss(hypothesis, Y)
      cost.backward()
      optimizer.step()

      avg_cost += cost / total_batch
  if epoch%10==0:
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
with torch.no_grad():
  
  test=pd.read_csv("test.csv",index_col = False).drop("시간", axis = 1) 
  wind=pd.DataFrame({'풍향': np.cos(np.pi*test["풍향"]/360)})
  test=test.drop("풍향",axis=1)
  test=pd.concat((test,wind),axis=1)
  test=np.array(test)

  test=Scaler.transform(test)
  test=torch.FloatTensor(test)

  hypothesis = model(test)
  predict1=torch.argmax(hypothesis,dim=1)
  submission=pd.read_csv('submission.csv', index_col=False,encoding='utf-8-sig')
  for i in range(len(predict1)):
    submission['PM10'][i]=int(predict1[i])

  submission['PM10']=submission['PM10'].astype(int)
  submission["PM10"]=submission["PM10"].replace([0,1,2,3],["좋음","보통","나쁨","매우나쁨"])
  

submission.to_csv('baseline.csv',index=False)
!kaggle competitions submit -c aqiprediction -f baseline.csv -m "Message"
