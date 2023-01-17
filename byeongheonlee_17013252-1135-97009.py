import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing
! pip uninstall kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6

! ls -lha kaggle.json
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json 
! kaggle competitions download -c ai-tomato
! unzip ai-tomato.zip
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(666)
torch.manual_seed(666)
if device =='cuda' :
    torch.cuda.manual_seed_all(666)
    
learning_rate = 0.1
training_epochs = 1000
batch_size = 10
xy_train = pd.read_csv("training_set.csv",header=None, skiprows=1)
xy_train
## data separate (월,일 정보도 사용)
xy_train[0] = xy_train[0]%10000/100

x_train = xy_train.loc[:,[i for i in xy_train.keys()[:-1]]]
y_train = xy_train[xy_train.keys()[-1]]

x_train

## data frame => numpy
x_train = np.array(x_train)
y_train = np.array(y_train)

## numpy => torch tensor
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
data_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)
linear1 = torch.nn.Linear(7,7,bias = True)
linear2 = torch.nn.Linear(7,1, bias= True)
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs) :

    avg_cost = 0

    for X, Y in data_loader :

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        h = model(X)

        cost = loss(h, Y)

        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    if epoch % 100 == 0 :
        print('Epoch {}, Cost : {}'.format(epoch,avg_cost))

print('Learning Finished')
test = pd.read_csv("test_set.csv",usecols=range(0,7),header=None, skiprows=1)

test[0] = test[0]%10000/ 100
test = test.loc[:,[i for i in test.keys()[:]]]
test
with torch.no_grad() :

    test = np.array(test)
    test = Scaler.transform(test)
    test = torch.from_numpy(test).float().to(device)   

    prediction = model(test)
## tensor => numpy
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('submit_sample.csv')
submit
for i in range(len(correct_prediction)) :
    submit['expected'][i] = correct_prediction[i].item()

submit
submit.to_csv('defence1.csv',index= False)
! kaggle competitions submit -c ai-tomato -f defence1.csv -m "baseline.0617.3"