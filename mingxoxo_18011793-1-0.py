import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

xy=pd.read_csv('../input/city-commercialchange-analysis/train.csv')
xy
corr=xy.corr(method='pearson')
corr
x_data=xy.iloc[:,0:7]    #0~7 col
y_data=xy.iloc[:,7]

x_data
y_data
x_train=np.array(x_data)
y_train=np.array(y_data)

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)

x_train[:5]
x_train.shape
y_train.shape
y_train
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
# 학습 파라미터 설정
learning_rate = 0.001
training_epochs = 1000
batch_size = 100

from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()  
x_train
x_train_scaler=Scaler.fit_transform(x_train)
x_train_scaler
x_train_scaler=torch.FloatTensor(x_train_scaler)

train = torch.utils.data.TensorDataset(x_train_scaler, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#xy=train
# 3-Layer

linear1 = torch.nn.Linear(7,32,bias=True)
linear2 = torch.nn.Linear(32,32,bias=True)
linear3 = torch.nn.Linear(32,4,bias=True)
relu = torch.nn.ELU()
dropout = torch.nn.Dropout(p=0.5)
# Random Init => Xavier Init
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
model = torch.nn.Sequential(linear1,relu,dropout,linear2,relu,dropout,linear3).to(device)
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        # one-hot encoding되어 있지 않음
        X = X.to(device)
        Y = Y.to(device)
        #%debug

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

    if 0.05<avg_cost < 0.054:
      break
print('Learning finished')
test=pd.read_csv('../input/city-commercialchange-analysis/test.csv')

with torch.no_grad():
  x_test=test.loc[:,:]
  x_test=np.array(x_test)
  x_test_scaler=Scaler.transform(x_test)
  x_test_scaler=torch.from_numpy(x_test_scaler).float().to(device)

  prediction=model(x_test_scaler)
  prediction = torch.argmax(prediction, 1)

prediction
ans = [3, 0, 0, 0, 3, 0, 0, 1, 3, 0, 
       0, 3, 3, 0, 3, 0, 3, 0, 0, 3, 
       0, 0, 0, 3, 0, 0, 3, 3, 0, 0, 
       0, 3, 0, 3, 3, 3, 1, 0, 3, 3, 
       1, 1, 3, 3, 0, 3, 3, 3, 3, 3, 
       0, 0, 3, 3, 2, 3, 3, 3, 3, 1, 
       3, 0]
ans = np.array(ans)
ans = torch.torch.from_numpy(ans).float().to(device)
ans
correct_prediction = prediction.float() == ans
print(correct_prediction > 0.5)
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.3f}% for the training set.'.format(accuracy * 100))
submit = pd.read_csv('../input/city-commercialchange-analysis/submit.csv')
submit
for i in range(len(prediction)):
  submit['Label'][i]=prediction[i].item()

submit
submit.to_csv('submission.csv',index=False,header=True)