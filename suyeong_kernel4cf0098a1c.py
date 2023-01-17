! pip uninstall -y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c 2020termproject-18011826
!unzip 2020termproject-18011826.zip
import numpy as np
import torch

xy_train=np.loadtxt('train_sweetpotato_price.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(1,6))
x_data=torch.from_numpy(xy_train[:,0:-1])
y_data=torch.from_numpy(xy_train[:,[-1]])

xy_test=np.loadtxt('test_sweetpotato_price.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(1,5))
test_x_data=torch.from_numpy(xy_test)

W=torch.zeros((4,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

optimizer=torch.optim.SGD([W,b],lr=0.00001)
epochs=100000

for epoch in range(epochs):
  hypothesis =x_data.matmul(W)+b
  cost=torch.mean((hypothesis-y_data)**2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 1000 ==0:
    print('epoch {}, cost {}'.format(
        epoch, cost.item()
    ))
prediction=test_x_data.matmul(W)+b

print(prediction)
import pandas as pd

submit=pd.read_csv('submit_sample.csv')

submit
for i in range (len(prediction)):
  submit['Expected'][i]=prediction[i].item()

submit
submit.to_csv('submit.csv',mode='w',index=False)

! kaggle competitions submit -c 2020termproject-18011826 -f submit.csv -m "submit"