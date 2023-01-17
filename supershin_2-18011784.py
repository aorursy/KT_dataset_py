# !pip uninstall -y kaggle
# !pip install --upgrade pip
# !pip install kaggle==1.5.6

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod=600 ~/.kaggle/kaggle.json

!kaggle competitions submit -c ai-project-life-environment
!unzip ai-project-life-environment
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as data
import torchvision.transforms as trasforms
import random
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
xy_train=np.loadtxt('./train data.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1,9))
x_data=torch.from_numpy(xy_train[:,0:-1]).to('cuda')
y_data=torch.from_numpy(xy_train[:,[-1]]).to('cuda')

xy_test=np.loadtxt('./train data.csv',delimiter=',', dtype=np.float32,skiprows=1,usecols=range(1,8))
test_x_data=torch.from_numpy(xy_test).to('cuda')
print(x_data)
print(y_data)
print(test_x_data)
class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()

    self.linear1 = nn.Linear(7, 512, bias = True)#기존 베이스라인 단일 레이어에서 멀티레이어로 확장
    self.linear2 = nn.Linear(512, 1, bias = True)

    self.relu = nn.ReLU()
    self.init_linear()

  def init_linear(self):
##w 초기값 설정
      for c in self.children():
          if isinstance(c, nn.Linear):
              nn.init.xavier_uniform_(c.weight)
              nn.init.constant_(c.bias, 0.)

  def forward(self, x):
    
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)

    return out
model = DNN().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss = nn.MSELoss().to('cuda')

epochs=5000
model.train()
for epoch in range(epochs) :
  
  hypothesis =model(x_data)
  cost=loss(hypothesis, y_data)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 1000==0:
    print("epoch:{}, cost:{}". format(epoch, cost.item()))
model.eval()
predict = model(test_x_data)
submit=pd.read_csv('./submit sample.csv')
for i in range(len(predict)):
  submit['Expected'][i]=predict[i].item()

submit
submit.to_csv('submit.csv',index = False, header = True)
