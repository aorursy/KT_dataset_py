import torch
import numpy as np
import pandas as pd
import torch.optim as optim
torch.manual_seed(1)
xy=pd.read_csv('2020AI_soil_train.csv', header=None,skiprows=1 )
x_data= xy.loc[:, 1:7]
y_data=xy[8]

x_data=np.array(x_data, dtype=float)
y_data= np.array(y_data, dtype=float)

y_data= y_data.reshape(-1,1)
x_data=torch.FloatTensor(x_data)
y_data=torch.FloatTensor(y_data)



from sklearn.preprocessing import*
scaler = MinMaxScaler()
x_data=scaler.fit_transform(x_data)
x_data=torch.FloatTensor(x_data)
print(x_data)

y_data=scaler.fit_transform(y_data)
y_data=torch.FloatTensor(y_data)
print(y_data)
W=torch.zeros((7,1),requires_grad=True )
b=torch.zeros(1,requires_grad=True)

optimizer= optim.SGD([W,b], lr=0.00058,momentum = 0.01)


nb_epochs=300000

for epoch in range (nb_epochs+1):
  hypothesis = x_data.matmul(W)+b
  cost= torch.mean((hypothesis-y_data)**2)


  optimizer.zero_grad()

  cost.backward()
  optimizer.step()

  if (epoch%100==0) :
    print(cost) 


x_testd=pd.read_csv('2020_soil_test.csv', header=None, skiprows=1)
x_test=x_testd.loc[:, 1:7]
x_test= np.array(x_test)
x_test= torch.FloatTensor(x_test)


print(x_test)



prediction= x_test.matmul(W)+b

submit1 = pd.read_csv('soil_submission.csv')
for i in range(len(prediction)):
  
  submit1['Expected'][i]=prediction[i].item()
print(submit1)
submit1.to_csv('baseline.csv',index=False,header=True)
! kaggle competitions submit -c 2020soil -f baseline.csv -m "baseline"