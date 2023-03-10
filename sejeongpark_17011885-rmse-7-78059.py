!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c taxi
!unzip taxi.zip
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
cab_type = {"Uber":0, "Lyft":1}

train['cab_type']= train['cab_type'].map(cab_type)
test['cab_type']= test['cab_type'].map(cab_type)
summary_mapping = {" Clear ":0," Drizzle ":1," Foggy ":2," Light Rain ":3," Mostly Cloudy ":4," Overcast ":5," Partly Cloudy ":6," Possible Drizzle ":7," Rain ":8}

train['short_summary'] = train['short_summary'].map(summary_mapping)
test['short_summary'] = test['short_summary'].map(summary_mapping)
icon_mapping = {" clear-day ":0," clear-night ":1," cloudy ":2," fog ":3," partly-cloudy-day ":4," partly-cloudy-night ":5," rain ":6}

train['icon'] = train['icon'].map(icon_mapping)
test['icon'] = test['icon'].map(icon_mapping)
train.info()
x_data = train.loc[0:,"cab_type":"precipIntensityMax"]
y_data = train.loc[0:,"price"]

x_data = np.array(x_data)
y_data = np.array(y_data)
scaler = MinMaxScaler()

x_data = scaler.fit_transform(x_data)
x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device).reshape(-1,1)

print(x_train[:5])
print(x_train.shape)
print(y_train[:5])
print(y_train.shape)
x_test = test.loc[0:,"cab_type":]
x_test = np.array(x_test)
x_test = scaler.transform(x_test)
x_test = torch.FloatTensor(x_test).to(device)

x_test[:3]
learning_rate = 5e-3
#1000
training_epochs = 500
batch_size =38
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1=torch.nn.Linear(19,19,bias=True)
linear2=torch.nn.Linear(19,1,bias=True)
#dropout=torch.nn.Dropout(p=0.3)
relu=torch.nn.ReLU()
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # ??????????????? ?????????
        optimizer.zero_grad()
        # Forward ??????
        hypothesis = model(X)
        # Error ??????
        cost = loss(hypothesis, Y)
        # Backparopagation
        cost.backward()
        # ????????? ??????
        optimizer.step()

        # ?????? Error ??????
        avg_cost += cost / total_batch
        
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
model(x_train)
with torch.no_grad():

  x_test=np.array(x_test)
  x_test=torch.from_numpy(x_test).float().to(device)
  predict=model(x_test)
predict
correct_prediction = predict.cpu().numpy().reshape(-1,1)
result = pd.read_csv('submission_form.csv')
for i in range(len(predict)):
  result['price'][i]=predict[i].item()
result
result.to_csv('submit.csv', index=False)