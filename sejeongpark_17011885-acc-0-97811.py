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
!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c lol-prediction
!unzip lol-prediction.zip
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # For Normalization
device = torch.device("cuda")
torch.manual_seed(484)
torch.cuda.manual_seed_all(484)
x_train = pd.read_csv('lol.x_train.csv', index_col=0)
y_train = pd.read_csv('lol.y_train.csv', index_col=0)
x_test = pd.read_csv('lol.x_test.csv', index_col=0)
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
train_set = TensorDataset(x_train, y_train)
data_loader = DataLoader(dataset=train_set,
                         batch_size=10000,
                         shuffle=True)
# DNN 모델 구축
linear1 = torch.nn.Linear(48, 32).to(device)
linear2=torch.nn.Linear(32,32).to(device)
linear3 = torch.nn.Linear(32, 1).to(device)

relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
dropout=torch.nn.Dropout(p=0.3)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)


model = torch.nn.Sequential(linear1, relu,dropout,
                            linear2,relu,dropout,
                            linear3, sigmoid)

model
cost = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 60
for epoch in range(1, epochs+1):
    avg_cost = 0
    total_batch = len(data_loader)

    for x, y in data_loader: 
         # batch loop
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        hypothesis = model(x)
        cost_val = cost(hypothesis, y)
        cost_val.backward()
        optimizer.step()

        avg_cost += cost_val
    
    avg_cost /= total_batch

    if epoch % 10 == 1 or epoch == epochs:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, avg_cost.item()))
with torch.no_grad(): # Don't Calculate Gradient
    model.eval()
    x_test = x_test.to(device)

    pred = model(x_test)
pred[pred>=0.5] = 1.0
pred[pred<=0.5] = 0.0
pred = pred.detach().cpu().numpy()
pred = pred.astype(np.uint32)
id=np.array([i for i in range(pred.shape[0])]).reshape(-1, 1).astype(np.uint32)
result=np.hstack([id, pred])

submit = pd.DataFrame(result, columns=['id', 'blueWins'])
submit.to_csv('submit.csv', index=False)
!kaggle competitions submit -c lol-prediction -f submit.csv -m "17011885"