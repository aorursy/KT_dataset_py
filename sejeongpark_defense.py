!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c carclassification
!unzip carclassification.zip
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
      torch.cuda.manual_seed_all(777)
train=pd.read_csv('car5_train.csv')
x_train=train.loc[:,[i for i in train.keys()[1:-1]]]
y_train=train[train.keys()[-1]]
Scaler=preprocessing.StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=40,
                                          shuffle=True,
                                          drop_last=True)
import torch.nn.functional as F
import torch.optim as optim

nb_class=8
nb_data=len(y_train)
l1 = torch.nn.Linear(8, 4)#딥러닝 모델로 수정
l2 = torch.nn.Linear(4, nb_class)
relu = torch.nn.LeakyReLU()
torch.nn.init.kaiming_uniform_(l1.weight)
torch.nn.init.kaiming_uniform_(l2.weight)
model = torch.nn.Sequential(l1, relu, l2).to(device)
model
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 
total_batch = len(data_loader)
model_history = []
err_history = []
for epoch in range(1, 1+300):
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
        avg_cost += cost
    avg_cost /= total_batch
    model_history.append(model)
    err_history.append(avg_cost)

    if epoch % 50 == 1 :
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
import matplotlib.pyplot as plt

plt.plot(err_history)
plt.show()
best_model = model_history[np.argmin(err_history)] 
test=pd.read_csv('car5_test.csv')
x_test=test.loc[:,[i for i in test.keys()[1:]]]
x_test=Scaler.transform(x_test) # Fit_transform 하면 안됨
x_test=np.array(x_test)
x_test=torch.FloatTensor(x_test)
with torch.no_grad():
    x_test = x_test.to(device)
    pred = best_model(x_test)
    predict=torch.argmax(pred,dim=1)

    print(predict.shape)
submit=pd.read_csv('car5_submit.csv')

submit[:5]
predict=predict.cpu().numpy().reshape(-1,1)

id=np.array([i for i in range(len(predict))]).reshape(-1,1)
result=np.hstack([id,predict])

submit=pd.DataFrame(result,columns=["Id","Category"])
submit.to_csv("baseline.csv",index=False,header=True)
submit