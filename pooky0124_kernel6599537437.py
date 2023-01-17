!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
import pandas as pd
import numpy as np
import random
import torch
import torch.optim as optim

! kaggle competitions download -c predict-number-of-asthma-patient
!unzip  predict-number-of-asthma-patient.zip
train_data = pd.read_csv('train_disease.csv', header=None, skiprows=1,usecols=range(0,6))
test_data = pd.read_csv('test_disease.csv',header=None, skiprows=1,usecols=range(1,5))

x_train_data = train_data.loc[:,1:4]
x_train_data = np.array(x_train_data)
x_train_data = torch.FloatTensor(x_train_data)

y_train_data = train_data.loc[:,5:5]
y_train_data = np.array(y_train_data)
y_train_data =torch.FloatTensor(y_train_data)

nb_epochs = 20000
w = torch.zeros((4,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.RMSprop([w,b], lr = 0.01, momentum=0.8)
nb_epochs = 200000

for epoch in range (nb_epochs+1):
  hypothesis = (x_train_data.matmul(w)+b)
  cost=torch.mean((hypothesis-y_train_data)**2)
  
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 10000 == 0 :
    print('Epoch : {}/{}, cost : {}'.format(
        epoch, nb_epochs, cost.item()
    ))
x_test = test_data.loc[:,1:4]
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test)
predict = x_test.matmul(w)+b

submit = pd.read_csv("submission.csv")
for i in range(len(predict)):
  submit['Expect'][i] = int(predict[i])
submit.dtypes
submit=submit.astype(int)
submit = submit.to_csv("submit.csv", mode='w', index=False)

!kaggle competitions submit -c predict-number-of-asthma-patients -f submit.csv -m "Message"