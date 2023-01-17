import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# 학습 파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c 2020-ai-exam-fashionmnist-2
!unzip 2020-ai-exam-fashionmnist-2.zip
import pandas as pd
import numpy as np
train_data=pd.read_csv('mnist_train_label.csv')
train_data
train_data=pd.read_csv('mnist_train_label.csv',header=None, usecols=range(0,785))
train_data
from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()
xy_train_data=train_data.to_numpy()


x_train_data=xy_train_data[:,1:]
x_train_data = Scaler.fit_transform(x_train_data)
y_train_data=xy_train_data[:,0]

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)
x_train_data.shape
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(784,10,bias=True)
torch.nn.init.normal_(linear1.weight)
# ======================================
# relu는 맨 마지막 레이어에서 빼는 것이 좋다.
# ======================================
# model = torch.nn.Sequential(linear1,relu,linear2,relu,linear3,relu).to(device) # 주의사항
model = torch.nn.Sequential(linear1).to(device)
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        # (1000, 1, 28, 28) 크기의 텐서를 (1000, 784) 크기의 텐서로 변형
        X = X.to(device)
        # one-hot encoding되어 있지 않음
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

print('Learning finished')
x_test = pd.read_csv('mnist_test.csv',header=None, usecols=range(0,784))
x_test = x_test.to_numpy()
x_test[:,0] = 0
x_test.shape
x_test = Scaler.transform(x_test)
# Test the model using test sets
with torch.no_grad():

  
  x_test_data=torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test_data)
  correct_prediction = torch.argmax(prediction, 1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
correct_prediction
submit=pd.read_csv('submission.csv')

for i in range(len(correct_prediction)):
  submit['Category'][i]=correct_prediction[i].item()
submit.to_csv('submit.csv',index=False,header=True)
!kaggle competitions submit -c 2020-ai-exam-fashionmnist-2 -f submit.csv -m "submit"