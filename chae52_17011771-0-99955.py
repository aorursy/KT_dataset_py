!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
from google.colab import files
#files.upload()
#kaggle이라는 폴더를 만든다
!mkdir -p ~/.kaggle
#폴더에 kaggle.json을 copy paste 한다
!cp kaggle.json ~/.kaggle/
#권한을 넣어준다
! chmod 600 ~/.kaggle/kaggle.json
#자세한 내용 출력
!ls -lha kaggle.json

#kaggle 버전 확인
!kaggle -v
!kaggle competitions download -c parkinglot
#위 코드를 통해 얻게되는 zip파일을 푼다
!unzip parkinglot.zip
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

train=np.loadtxt('train.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,14)) 

xtrain=torch.from_numpy(train[:,0:-1])
ytrain=torch.from_numpy(train[:,[-1]])

scaler = preprocessing.StandardScaler() #스케일 조정
xtrain = scaler.fit_transform(xtrain)

xtrain = torch.FloatTensor(xtrain)
ytrain = torch.FloatTensor(ytrain)
dataset = TensorDataset(xtrain, ytrain)
dataloader=DataLoader(dataset, batch_size=13, shuffle=True)

model = nn.Sequential(
    nn.Linear(13, 128),
    nn.Linear(128,1),
    nn.Sigmoid()
)

torch.manual_seed(0)

optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    x_data, y_data = samples

    H = model(x_data)
    cost = -(y_data * torch.log(H) + (1-y_data)*torch.log(1-H)).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

  if epoch % 10 == 0:
    prediction = H >= torch.FloatTensor([0.5])
    correct_prediction = prediction.float() == y_data
    accuracy = correct_prediction.sum().item() / len(correct_prediction)
    print('Epoch {} Cost {} Accuracy {:.2f}%'.format(epoch, cost.item(), accuracy*100))
test=np.loadtxt('test.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,13))  
xtest=torch.from_numpy(test)
xtest = scaler.transform(xtest)
xtest = torch.FloatTensor(xtest)

H = model(xtest)

predict = H >= torch.FloatTensor([0.5])

submit=pd.read_csv('submission.csv')

for i in range(len(predict)):
  submit['Expected'][i]=int(predict[i])

submit=submit.astype(int)
submit.to_csv('sub.csv',index=False) 
submit
!kaggle competitions submit -c parkinglot -f sub.csv -m "submit"



