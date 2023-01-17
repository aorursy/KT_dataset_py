!pip install kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions list
!kaggle competitions download -c 2020-ml-w1p3
!ls
!unzip
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
device = torch.device("cuda")
#데이터
xy_data = pd.read_csv('/content/2020.AI.cancer-train.csv')
x_test = pd.read_csv('/content/2020.AI.cancer-test.csv')
submit = pd.read_csv('/content/2020.AI.cancer-sample-submission.csv')
xy_data = np.array(xy_data)
x_train = torch.FloatTensor(xy_data[:,1:-1]).to(device)
y_train = torch.FloatTensor(xy_data[:,0]).to(device)
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test[:,:-1]).to(device)
x_train.shape
W = torch.zeros((30,1)).to(device).detach().requires_grad_(True)
b = torch.zeros((398,1)).to(device).detach().requires_grad_(True)
optimizer= optim.SGD([W,b], lr=1e-3, momentum=0.9)
nb_epochs = 10000
from torch.nn import BCELoss
import torch.nn.functional as F
loss =BCELoss()
for epoch in range(nb_epochs + 1):
  hypothesis = torch.sigmoid(x_train.matmul(W)+b)
  cost = loss(hypothesis, y_train)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  if epoch%1000==0:
    print('cost = {}'.format(cost.item()))
hypothesis = torch.sigmoid(x_test.matmul(W)+b)
predict = hypothesis>=0.5
for i in range(len(predict)):
  submit['diagnosis'][i]=predict[i]
submit=submit.astype(np.int32)
submit.to_csv('submit.csv', mode='w', header= True, index= False)
!kaggle competitions submit -c 2020-ml-w1p3 -f submit.csv -m "Message"