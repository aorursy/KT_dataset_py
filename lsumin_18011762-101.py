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
!kaggle competitions download -c projectmosquito
!unzip projectmosquito.zip
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd

from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train_data=pd.read_csv('train.csv')
train_data
test_data=pd.read_csv('test.csv')
test_data
train_data["year"]=train_data["year"]%10000/100
test_data["year"]=test_data["year"]%10000/100

x_train_data = train_data.loc[:,[i for i in train_data.keys()[:-1]]]
y_train_data = train_data[train_data.keys()[-1]]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
# 학습 파라미터 설정
learning_rate = 0.005
training_epochs = 100
batch_size = 1
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(4,128,bias=True)
linear2 = torch.nn.Linear(128,128,bias=True)
linear7 = torch.nn.Linear(128,1,bias=True)
relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.kaiming_uniform_(linear7.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2,relu, 
                            linear7).to(device)

loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)

for epoch in range(training_epochs+1):
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

    if epoch%10==0:
      print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
# Test the model using test sets
with torch.no_grad():
  model.eval()

  x_test_data=test_data.loc[:,[i for i in test_data.keys()[:]]]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submit_sample.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()

submit
submit.to_csv('18011762.csv',index=False,header=True)

! kaggle competitions submit -c projectmosquito -f 18011762.csv -m "18011762"