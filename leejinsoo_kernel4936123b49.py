import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing

!kaggle competitions download -c 2020soil
Scaler=preprocessing.StandardScaler()
!unzip 2020soil.zip
learning_rate = 1e-4
training_epoches = 100
batch_size = 100
Scaler = preprocessing.StandardScaler()
#drop_prob=0.3
device = torch.device('cuda')
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)
train = pd.read_csv('2020AI_soil_train.csv', header=None, skiprows=1,usecols=range(1,9))
test = pd.read_csv('2020_soil_test.csv', header = None, skiprows=1, usecols=range(1,8))
x_train = train.loc[:,1:7]
y_train = train.loc[:,8:8]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = Scaler.fit_transform(x_train)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last=True)
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
linear1 = torch.nn.Linear(7,64, bias = True) # feature
linear2 = torch.nn.Linear(64,64, bias = True)
linear3 = torch.nn.Linear(64,128, bias = True)
linear4 = torch.nn.Linear(128,512, bias = True)
linear5 = torch.nn.Linear(512,512, bias = True)
linear6 = torch.nn.Linear(512,512, bias = True)
linear7 = torch.nn.Linear(512,256, bias = True)
linear8 = torch.nn.Linear(256,128, bias = True)
linear9 = torch.nn.Linear(128,64, bias = True)
linear10 = torch.nn.Linear(64,8, bias = True)
linear11 = torch.nn.Linear(8,4, bias = True)
linear12 = torch.nn.Linear(4,1, bias = True)
mish = Mish()
torch.nn.init.kaiming_uniform_(linear1.weight)
torch.nn.init.kaiming_uniform_(linear2.weight)
torch.nn.init.kaiming_uniform_(linear3.weight)
torch.nn.init.kaiming_uniform_(linear4.weight)
torch.nn.init.kaiming_uniform_(linear5.weight)
torch.nn.init.kaiming_uniform_(linear6.weight)
torch.nn.init.kaiming_uniform_(linear7.weight)
torch.nn.init.kaiming_uniform_(linear8.weight)
torch.nn.init.kaiming_uniform_(linear9.weight)
torch.nn.init.kaiming_uniform_(linear10.weight)
torch.nn.init.kaiming_uniform_(linear11.weight)
torch.nn.init.kaiming_uniform_(linear12.weight)
model = torch.nn.Sequential(linear1,mish,
                            linear2,mish,
                            linear3,mish,
                            linear4,mish,
                            linear5,mish,
                            linear6,mish,
                            linear7,mish,
                            linear8,mish,
                            linear9,mish,
                            linear10,mish,
                            linear11,mish,
                            linear12
                            ).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)
#model.train()
for epoch in range(training_epoches):
  avg_cost = 0

  for X, Y in data_loader:

    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()
    avg_cost += cost / total_batch
  
  print('Epoch:','%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
print('Learning finshed')
x_test = test.loc[:,1:7]
x_test = np.array(x_test)
x_test = Scaler.transform(x_test)
with torch.no_grad():
  #model.eval()

  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('soil_submission.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i] = correct_prediction[i].item()

submit
submit.to_csv('1차시도.csv', mode='w', index = False)
! kaggle competitions submit -c 2020soil -f 1차시도.csv -m "Message"