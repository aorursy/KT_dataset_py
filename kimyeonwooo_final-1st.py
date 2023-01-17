!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c aiseaweather
!unzip aiseaweather.zip
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from torch.utils.data import  TensorDataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.0001
training_epochs = 5000
batch_size = 100
xy_train = pd.read_csv('train_wave.csv', header = None, skiprows=1, usecols=range(2, 13))
x_data = xy_train.loc[:1705, 1:11]
y_data = xy_train.loc[:1705, [12]]
x_data = np.array(x_data)
y_data = np.array(y_data)

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device) 
train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)
linear1 = torch.nn.Linear(10, 5,bias=True)
linear2 = torch.nn.Linear(5, 5,bias=True)
linear3 = torch.nn.Linear(5, 1,bias=True)
relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []
model_history = []
err_history = []

total_batch = len(data_loader)
model.train()

for epoch in range(training_epochs + 1):
  avg_cost = 0

  for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = torch.mean((hypothesis - Y) ** 2)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch

  model_history.append(model)
  err_history.append(avg_cost)
  
  if epoch % 10 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
plt.plot(losses)
plt.plot(err_history)
plt.show()
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test_wave.csv', header = None, skiprows=1, usecols = range(2, 12))
x_data = xy_test.loc[:, 1:11]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()  # 주의사항 (dropout=False)
    
    predict = best_model(x_test)
predict
submit = pd.read_csv('submit_sample.csv')
submit['Expected'] = submit['Expected'].astype(float)
for i in range(len(predict)):
  submit['Expected'][i] = predict[i]
submit.to_csv('submit.csv', index = False, header = True)
!kaggle competitions submit -c aiseaweather -f submit.csv -m "gg"
