!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c predict-number-of-asthma-patient
!unzip predict-number-of-asthma-patient.zip
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from sklearn import preprocessing
device='cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device=='cuda':
  torch.cuda.manual_seed_all(777)

learning_rate = 0.0001
training_epochs=200
batch_size=1
scaler = preprocessing.MinMaxScaler()
train_data = pd.read_csv('train_disease.csv', skiprows=1, header = None, usecols=range(1,6))
test_data = pd.read_csv('test_disease.csv', skiprows=1, header=None, usecols=range(1,5))
train_data
x_train_data = train_data.loc[:,0:4]
y_train_data = train_data.loc[:,5]
x_train_data[3] = 1010-x_train_data[3]
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = scaler.fit_transform(x_train_data)

x_train_data = torch.FloatTensor(x_train_data).to(device)
y_train_data = torch.FloatTensor(y_train_data).to(device)
x_train_data
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(4,512, bias=True)
linear2 = torch.nn.Linear(512,512, bias=True)
linear3 = torch.nn.Linear(512,512, bias=True)
linear4 = torch.nn.Linear(512,64, bias=True)
linear5 = torch.nn.Linear(64,64, bias=True)
linear6 = torch.nn.Linear(64, 8, bias=True)
linear7 = torch.nn.Linear(8,4, bias=True)
linear8 = torch.nn.Linear(4,1, bias=True)
relu = torch.nn.ReLU()
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)
torch.nn.init.xavier_uniform_(linear8.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4,relu,
                            linear5,relu,
                            linear6,relu,
                            linear7,relu,
                            linear8).to(device)
loss = torch.nn.MSELoss().to(device) #regression 문제라 MSELoss 쓴다
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
for epoch in range(training_epochs):
  avg_cost = 0

  for X, Y in data_loader:
    X=X.to(device)
    Y=Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/total_batch

  print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('learning finished')
test_data[3] = 1010-test_data[3]
test_data
with torch.no_grad():
  x_test_data = test_data.loc[:,:]
  x_test_data = np.array(x_test_data)
  x_test_data = scaler.fit_transform(x_test_data)
  x_test_data = torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)

submit = pd.read_csv('submission.csv')
for i in range(len(correct_prediction)):
  submit['Expect'][i]=correct_prediction[i].item()

submit
submit.to_csv('submission.csv',index=False,header=True)

!kaggle competitions submit -c predict-number-of-asthma-patient -f submission.csv -m "Message"