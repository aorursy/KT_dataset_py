import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

! pip uninstall -y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6 
! ls -lha kaggle.json

! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v


! kaggle competitions download -c ai-tomato
! unzip ai-tomato.zip
xy_train = pd.read_csv("training_set.csv")
xy_train
## data separate (월,일 정보도 사용)
xy_train["date"] = xy_train["date"]%10000/100
x_train = xy_train.loc[:,[i for i in xy_train.keys()[:-1]]]
y_train = xy_train[xy_train.keys()[-1]]
## data frame => numpy
x_train = np.array(x_train)
y_train = np.array(y_train)

## numpy => torch tensor
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

y_train = y_train.unsqueeze(dim=1)
y_train

w = torch.zeros((7,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
optimizer = optim.SGD([w,b],lr=1e-3,momentum=0.9)
epochs = 30001

for epoch in range(epochs) :

    h = x_train.matmul(w)+b

    cost = torch.mean((h-y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 3000==0 :
        print('Epoch : {}, Cost : {}'.format(epoch, cost.item()))
test = pd.read_csv("test_set.csv",usecols=range(0,7))
test
test["date"] = test["date"]%10000/100
test
## data frame => numpy
test = np.array(test)
## numpy => torch Tensor
test = torch.FloatTensor(test)
prediction = test.matmul(w) +b
print(prediction[:5])

df = pd.read_csv('submit_sample.csv')

print(df)

for i in range(len(prediction)) :
    df['expected'][i] = prediction[i]

print(df)

df.to_csv('baseline.csv',index=False)

! kaggle competitions submit -c ai-tomato -f baseline.csv -m "baseline"