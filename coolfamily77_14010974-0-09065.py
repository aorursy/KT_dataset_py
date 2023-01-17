!pip uninstall --y kaggle
!pip install --upgrade pip
!pip  install kaggle==1.5.6

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle -v
!kaggle competitions download -c solarenergy-meteorologicalphenomenon2
!unzip solarenergy-meteorologicalphenomenon2.zip
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import random

from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
learning_rate = 1e-3
training_epochs = 900
batch_size = 50
Scaler = preprocessing.StandardScaler()
train = pd.read_csv('Solar_TrainData_3.csv',header=None,skiprows=1,usecols=range(0,9))
train = train.dropna()
print(train.head(10))
print(train.info())
test = pd.read_csv('Solar_TestData_2.csv',header=None,skiprows=1,usecols=range(0,8))

print(test.head(10))
print(test.info())
x_train = train.iloc[:,1:7]
y_train = train.iloc[:,[-1]]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
print(x_train.shape)
print(y_train.shape)
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)
linear1 = torch.nn.Linear(6,32, bias = True) # feature
linear2 = torch.nn.Linear(32,32, bias = True)
linear3 = torch.nn.Linear(32,32, bias = True)
linear4 = torch.nn.Linear(32,16, bias = True)
linear5 = torch.nn.Linear(16,16, bias = True)
linear6 = torch.nn.Linear(16,16, bias = True)
linear7 = torch.nn.Linear(16,8, bias = True)
linear8 = torch.nn.Linear(8,8, bias = True)
linear9 = torch.nn.Linear(8,8, bias = True)
linear10 = torch.nn.Linear(8,1, bias = True)

mish = Mish()
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)
torch.nn.init.xavier_uniform_(linear8.weight)
torch.nn.init.xavier_uniform_(linear9.weight)
torch.nn.init.xavier_uniform_(linear10.weight)
model = torch.nn.Sequential(linear1,mish,
                            linear2,mish,
                            linear3,mish,
                            linear4,mish,
                            linear5,mish,
                            linear6,mish,
                            linear7,mish,
                            linear8,mish,
                            linear9,mish,
                            linear10).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0
    for X,Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch
    print('epoch ','%04d' % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost))
print('Learning finished..!')
with torch.no_grad():
  x_test = test.iloc[:,1:7]
  x_test = np.array(x_test)

  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
MAKE = pd.read_csv('Solar_TestData_2.csv', header = None, skiprows= 1) 
submit = pd.read_csv('Solar_SubmitForm_2.csv')
submit
for i in range(len(correct_prediction)):
  submit['Predict'][i] = correct_prediction[i].item()

submit['YYYY/MM/DD'] = MAKE[0]
submit
submit.to_csv('result.csv', mode='w', index = False)
! kaggle competitions submit -c solarenergy-meteorologicalphenomenon2 -f result.csv -m ""