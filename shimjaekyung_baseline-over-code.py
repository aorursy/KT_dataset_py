!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls -lha kaggle.json

!chmod 600 ~/.kaggle/kaggle.json
!ls -lha kaggle.json
! kaggle competitions download -c rainyseason
!unzip rainyseason.zip
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
torch.manual_seed(1)
xy_train = np.loadtxt('rainyseason-train.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(4,13))

x_train = torch.from_numpy(xy_train[:,1:])

y_data = xy_train[:,[0]].squeeze()
y_train = torch.LongTensor(y_data)

xy_test = np.loadtxt('rainyseason-test.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(5,13))
test_x_data = torch.from_numpy(xy_test)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=50,
                                          shuffle=True,
                                          drop_last=True)
nb_class=8
nb_data=len(y_train)
l1 = torch.nn.Linear(8, 2)       #딥러닝 모델 구현
l2 = torch.nn.Linear(2, nb_class)
relu = torch.nn.ReLU()
torch.nn.init.xavier_uniform_(l1.weight)
torch.nn.init.xavier_uniform_(l2.weight)
model = torch.nn.Sequential(l1, relu, l2).to(device)
model
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

total_batch = len(data_loader)
model_h = []
error_h = []
for epoch in range(1, 1+2000):
  avg_cost = 0

  for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

   
     
  optimizer.zero_grad()
  hypothesis = model(X)
  
  cost = loss(hypothesis, Y)
  cost.backward()
  optimizer.step()
  avg_cost += cost
  avg_cost /= total_batch
  model_h.append(model)
  error_h.append(avg_cost)

  if epoch % 50 == 1 :
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
best_model = model_h[np.argmin(error_h)]
x_test=torch.FloatTensor(test_x_data)
with torch.no_grad():
    x_test = x_test.to(device)
    pred = best_model(x_test)
    predict=torch.argmax(pred,dim=1)

    print(predict.shape)
predict
submit=pd.read_csv('rainyseason-sample.csv')
submit
predict=predict.cpu().numpy().reshape(-1,1)

id=np.array([i for i in range(len(predict))]).reshape(-1,1)
result=np.hstack([id,predict])

submit=pd.DataFrame(result,columns=["Id","RainySeason"])
submit.to_csv("submit.csv",index=False,header=True)
submit
!kaggle competitions submit -c rainyseason -f submit.csv -m "15011139 심재경"