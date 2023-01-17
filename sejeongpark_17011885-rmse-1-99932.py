!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c forrest-gump-1994
!unzip forrest-gump-1994.zip
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # For Normalization
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train= pd.read_csv("kaggle_train.csv")
x_train= train.iloc[:,1:-1]
y_train= train.iloc[:,[-1]]
train
x_train= x_train.to_numpy()
y_train = y_train.to_numpy()

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
learning_rate = 0.01
#1000
training_epochs = 100
batch_size =50
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear=torch.nn.Linear(9718,1,bias=True)
#dropout=torch.nn.Dropout(p=0.3)
relu=torch.nn.ReLU()
torch.nn.init.xavier_uniform_(linear.weight)
model = torch.nn.Sequential(linear).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # Forward 계산
        hypothesis = model(X)
        # Error 계산
        cost = loss(hypothesis, Y)
        # Backparopagation
        cost.backward()
        # 가중치 갱신
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
model(x_train)
x_test = pd.read_csv("kaggle_test.csv",
                        encoding = 'utf,8',
                        )
x_test.head()
x_test =test_data.to_numpy()
x_test = torch.FloatTensor(x_test)
with torch.no_grad():
  model.eval()
  x_test=test_data.iloc[:,1:]
  x_test=np.array(x_test)
  x_test=torch.from_numpy(x_test).float().to(device)
  predict=model(x_test)
correct_prediction = predict.cpu().numpy().reshape(-1,1)
predict
result = pd.read_csv('submit_sample.csv')
for i in range(29):
  result['result[Forrest Gump (1994)]'][i]=predict[i].item()
result
result.to_csv('submit.csv', index=False)