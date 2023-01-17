import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
if torch.cuda.is_available() is True:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
pd_train = pd.read_excel("./drive/My Drive/traindata.xlsx")
# pd_train = pd.read_excel("testtest.xlsx")
pd_train.info()
pd_train.iloc[:,0].sort_values()
tmp = []

for i, v in enumerate(pd_train.iloc[:,0].tolist()):

  if v == pd_train.iloc[:,0].max():
    tmp.append(i)

pd_data = pd_train.drop(tmp)
pd_data = pd_data.drop(labels = ['binnedlnc'], axis=1)

pd_data.dropna(axis=0, inplace = True)
pd_x = pd_data.iloc[:,:-1]
pd_y = pd_data.iloc[:,-1]
##학습 데이터셋 정규화
for col in pd_x:
  # import pdb;pdb.set_trace()
  mean = pd_x[col].mean()
  std = pd_x[col].std()
  pd_x[col] = (pd_x[col]- mean) / std
pd_x
x_train = torch.FloatTensor(np.array(pd_x)).to(device);
y_train = torch.FloatTensor(np.array(pd_y)).to(device);

print(x_train);
print(y_train);
print(x_train.shape);
print(y_train.shape);
torch.manual_seed(777);
torch.cuda.manual_seed_all(777);


linear1 = nn.Linear(29, 256, bias = True)
linear2 = nn.Linear(256,512, bias = True)
linear3 = nn.Linear(512, 1, bias = True)

drop = nn.Dropout(p=0.5)


relu = torch.nn.ReLU()

model = nn.Sequential(linear1, relu, drop, linear2, relu, drop, linear3).to(device)

## 초기값 init
for layer in model.children():

  if isinstance(layer, nn.Linear):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0.)

lr = 1e-3
Epochs = 4900

optimizer = optim.Adam(model.parameters(), lr = lr)
loss = nn.MSELoss().to(device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 500, factor= 0.1, verbose = True)

for epoch in range(Epochs+1):
  model.train()
  optimizer.zero_grad()
  H = model(x_train)
  cost = torch.sqrt((loss(H , y_train)))
  cost.backward()
  optimizer.step()
  scheduler.step(cost)
  if epoch % 100 == 0:
    print("train - Epochs : {}/{}  avg_cost = {:.6f}".format(epoch, Epochs, cost.item()))


##model test
pd_test = pd.read_excel("./drive/My Drive/testdata.xlsx")

pd_data2 = pd_test.iloc[:,:]
pd_data2 = pd_data2.drop(labels = ['binnedlnc'], axis=1)
pd_data2
pd_data2.info()
for col in pd_data2:
  mean = pd_data2[col].mean()
  std = pd_data2[col].std()

  pd_data2[col] = (pd_data2[col] - mean) / std
pd_data2

x_test = torch.FloatTensor(np.array(pd_data2)).to(device)

print(x_test.shape)
with torch.no_grad():
  model.eval()
  predict =model(x_test)

DeathRate = predict.to('cpu').detach().numpy().reshape(-1,1)
ID = np.array([i for i in range(len(y_test))]).reshape(-1,1)
result = np.hstack((ID, DeathRate));

df = pd.DataFrame(result, columns = ['ID', 'DeathRate'])
df.to_csv("baseline.csv",header = True, index = False)
