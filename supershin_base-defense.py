!pip uninstall --y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6



!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle

!chmod 600 ~/.kaggle/kaggle.json
if torch.cuda.is_available() is True:

  device = torch.device("cuda")

else:

  device = torch.device("cpu")
import torch

import numpy as np

import pandas as pd

import torch.optim as optim

import torch.nn as nn
pd_train = pd.read_excel("./drive/My Drive/traindata.xlsx")
##model test

pd_test = pd.read_excel("./drive/My Drive/testdata.xlsx")
tmp = []





for i, v in enumerate(pd_train.iloc[:,0].tolist()):

  if v == pd_train.iloc[:,0].max():

    tmp.append(i)
pd_data = pd_train.drop(tmp, axis = 0)

pd_data = pd_data.drop(labels = ['binnedlnc','PctMarriedHouseholds','PctOtherRace','AvgHouseholdSize','PercentMarried', 'MedianAge'], axis=1)



pd_data.dropna(axis=0, inplace = True)



pd_data2 = pd_test.iloc[:,:]

pd_data2 = pd_data2.drop(labels = ['binnedlnc','PctMarriedHouseholds','PctOtherRace','AvgHouseholdSize','PercentMarried', 'MedianAge'], axis=1)

pd_data2
pd_x = pd_data.iloc[:,:-1]

pd_y = pd_data.iloc[:,-1]
for col in pd_x:

  min = pd_x[col].min()

  max = pd_x[col].max()

  pd_x[col] = (pd_x[col]- min) / (max-min)

  pd_data2[col] = (pd_data2[col]-min)/(max-min)
x_train = torch.FloatTensor(np.array(pd_x)).to(device);

y_train = torch.FloatTensor(np.array(pd_y)).to(device);



x_test = torch.FloatTensor(np.array(pd_data2)).to(device)

y_test = torch.FloatTensor(np.array(pd_testy.iloc[:,-1]).reshape(-1,1)).to(device)



print(x_train.shape);

print(y_train.shape);

print(x_test.shape);

print(y_test.shape);
from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 100)

print(X_train.shape, Y_train.shape)

print(X_valid.shape, Y_valid.shape)

datazip = torch.utils.data.TensorDataset(X_train, Y_train)

train_loader = torch.utils.data.DataLoader(dataset = datazip,

                                      batch_size = 200,

                                      shuffle = True,

                                      drop_last = True)
torch.manual_seed(777);

torch.cuda.manual_seed_all(777);



linear1 = nn.Linear(24, 256, bias = True)

linear2 = nn.Linear(256,512, bias = True)

linear3 = nn.Linear(512, 1, bias = True)



drop = nn.Dropout(p=0.5)



relu = torch.nn.ReLU()



model = nn.Sequential(linear1, relu, drop, linear2, relu, drop, linear3).to(device)

# model = nn.Sequential(linear1, relu, drop, linear2, relu, drop, linear3).to(device)

for layer in model.children():

  # import pdb;pdb.set_trace()

  if isinstance(layer, nn.Linear):

    nn.init.xavier_uniform_(layer.weight)

    nn.init.constant_(layer.bias, 0.)

lr = 1e-3

Epochs = 3000



optimizer = optim.Adam(model.parameters(), lr = lr)

loss = nn.MSELoss().to(device)
best_cost = 30



for epoch in range(Epochs):

  model.train()



  for X,Y in train_loader:

    optimizer.zero_grad()

    H = model(X)



    cost = torch.sqrt(loss(H , Y))

    cost.backward()

    optimizer.step()



  with torch.no_grad():

    model.eval()



    cost2 = torch.sqrt(((model(X_valid) - Y_valid)**2).mean())

    if best_cost > cost2:

      best_cost = cost2

      print("new_best_model, epoch : {:4d}".format(epoch))

      torch.save(model, "./best_model4.ptr")



    if epoch % 100 == 0:

        print("test - Epochs : {}/{}  avg_cost = {:.6f}".format(epoch, Epochs, cost2.item()))

model = torch.load("./best_model4.ptr")

model.eval()



predict = model(x_test)

predict.shape
tmp = predict.to('cpu').detach().numpy()

ID = np.array([i for i in range(len(y_test))]).reshape(-1,1)

result = np.hstack((ID, tmp))
df = pd.DataFrame(result, columns=['ID', 'DeathRate'])

df.to_csv("submit_example.csv",index=False,header=True)
!kaggle competitions submit -c 2020-ai-termproject -f submit_example.csv -m "Message"