! pip uninstall --y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c star-classifier
! unzip star-classifier.zip
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
import pandas as pd
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu' #GPU 연결되어 있으면 GPU 쓰고 아니면 CPU쓰기

# 디버깅 편리하게 하기 위해서 seed 고정시키기
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train = pd.read_csv('star_train.csv',header=None, skiprows=1, usecols=range(1,8))
train
test = pd.read_csv('star_test.csv', header=None, skiprows=1, usecols=range(1,7))
test
x_train = train.loc[:,1:6] #크게 다른 건 없지만, 데이터 파싱하는 부분이 달랐고
y_train = train.loc[:,7:7]

x_train = np.array(x_train)
y_train = np.array(y_train).squeeze(1) 
# 어떤 식으로 하셔서 된 건지 모르겠지만, 이 부분에서 squeeze로 matrix 스케일 맞춰줌
# cross_entrophy loss 실행 부분에서 multi~ 오류 해결 위해 넣음

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
learning_rate = 0.0001 #lr
training_epoches = 20000 #epoches, running
batch_size = 1000 # 문제 몇 번 풀고 정답 체크
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last=False)
linear1 = torch.nn.Linear(6,32, bias = True)
linear2 = torch.nn.Linear(32,32, bias = True)
linear3 = torch.nn.Linear(32,16, bias = True)
linear4 = torch.nn.Linear(16,16, bias = True)
linear5 = torch.nn.Linear(16,16, bias = True)
linear6 = torch.nn.Linear(16,8, bias = True)
linear7 = torch.nn.Linear(8,8, bias = True)
linear8 = torch.nn.Linear(8,6, bias = True)
relu = torch.nn.ReLU()
# # Random Init => Xavier Init
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
                            linear8).to(device) #순서 지키기!, 맨 마지막은 relu 제외하기
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(training_epoches):
  avg_cost = 0

  for X,Y in data_loader:

    X = X.to(device)
    Y = Y.to(device) 

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
    
  print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{: .9f}'.format(avg_cost))
print('learning finished')
with torch.no_grad():
  x_test = test
  x_test = np.array(x_test)
  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
  correct_prediction = torch.argmax(prediction,1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('star_sample.csv')
submit
for i in range(len(correct_prediction)):
  submit['Label'][i] = correct_prediction[i].item()

submit
submit.to_csv('submission.csv', index=False, header=True)
! kaggle competitions submit -c star-classifier -f submission.csv -m "Message"