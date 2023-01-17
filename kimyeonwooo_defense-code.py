# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

drop_prob = 0.5
device = 'cuda'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train=pd.read_csv("new_train.csv")
x_train=train.iloc[:,1:]
y_train=train.iloc[:,[0]]

x_train=np.array(x_train)
y_train=np.array(y_train)

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
test=pd.read_csv("new_test.csv")
y_train=y_train.squeeze()
y_train.shape
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          drop_last=True)
l1 = torch.nn.Linear(5, 10)
l2 = torch.nn.Linear(10,10)
l3 = torch.nn.Linear(10,10)
l4 = torch.nn.Linear(10,2)

relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(l1.weight)
torch.nn.init.xavier_uniform_(l2.weight)
torch.nn.init.xavier_uniform_(l3.weight)
torch.nn.init.xavier_uniform_(l4.weight)

bn1=torch.nn.BatchNorm1d(10)
bn2=torch.nn.BatchNorm1d(10)
bn3=torch.nn.BatchNorm1d(10)

model = torch.nn.Sequential(l1,bn1, relu,l2,bn2,relu,l3,bn3,relu,l4).to(device)
model
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
total_batch = len(data_loader)
model_h = []
error_h = []
for epoch in range(1, 301):
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

  if epoch % 10 == 0 :
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
best_model = model_h[np.argmin(error_h)]
test=np.array(test)

test=torch.FloatTensor(test)

with torch.no_grad():
    test = test.to(device)
    pred = best_model(test)
    predict=torch.argmax(pred,dim=1)

    print(predict.shape)
submit = pd.read_csv('submit_sample.csv')

predict=predict.cpu().numpy().reshape(-1,1)

id=np.array([i for i in range(len(predict))]).reshape(-1,1)
result=np.hstack([id,predict])

submit=pd.DataFrame(result,columns=["id","index"])
submit.to_csv("defense.csv",index=False,header=True)