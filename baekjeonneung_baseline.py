import numpy as np
import pandas as pd
import torch
import torch.optim as optim
torch.manual_seed(1)

!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle -v
!unzip 18011854kbopredicton.zip
xy=np.loadtxt('kbo_train.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(0,9))
x_data=torch.from_numpy(xy[:,0:-1])
y_data=torch.from_numpy(xy[:,[-1]])
test=np.loadtxt('kbo_test.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,8))
x_test=torch.from_numpy(test)
w=torch.zeros((8,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
optimizer=optim.SGD([w, b], lr=1e-9)
nb_epochs=300000

for epoch in range(nb_epochs+1):
  hypo=x_data.matmul(w)+b
  cost=torch.mean((hypo-y_data)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  if(epoch%30000==0):
    print('epo : {:4d} cost : {:.6f}'.format(epoch, cost.item()
    ))
predict=x_test.matmul(w)+b
print(x_test)
print(predict)
submit=pd.read_csv('submit_sample.csv')
for i in range(len(predict)):
  submit['Expected'][i]=predict[i].item()
submit.to_csv('baseline.csv',mode='w',index=False)
from google.colab import files

files.download('baseline.csv') 