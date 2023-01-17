!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c 2020-ai-exam-fashionmnist-1
!unzip 2020-ai-exam-fashionmnist-1
import pandas as pd
import numpy as np
import torch
train_csv = pd.read_csv('mnist_train_label.csv')
test_csv = pd.read_csv('mnist_test.csv') 
train_csv.head()
train_csv.describe()
train_L = train_csv.iloc[:,0]
train_D = train_csv.drop('9',axis =1)
len(train_D)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
device= 'cpu'
#단층. (28*28, 10 ) class가 0~9까지니깐 
l1 = torch.nn.Linear(train_D.shape[1], 10, bias = True)
#WEIGHT 초기화 해주기 
torch.nn.init.xavier_uniform_(l1.weight)

model = torch.nn.Sequential(l1).to(device)
#crossentropy사용, 최적화 SGD 
lr = 0.1
epoch = 15

loss = torch.nn.CrossEntropyLoss().to(device) # to gpu
optimizer = torch.optim.SGD(model.parameters(), lr=lr) # 0.49
train_D = torch.FloatTensor(np.array(train_D))
train_L = torch.LongTensor(np.array(train_L))
train_L
for stop in range(epoch):
    
    # 그래디언트 초기화
    optimizer.zero_grad()
    # Forward 계산
    hypothesis = model(train_D)
    # Error 계산
    cost = loss(hypothesis, train_L)
    # Backward 계산 
    cost.backward()
    # 가중치 갱신
    optimizer.step()
    print(stop, cost.item())

test_D= torch.FloatTensor(np.array(test_csv))
with torch.no_grad():
  model.eval()
  pred= model(test_D.to(device))

pred_list= torch.argmax(pred, 1).cpu()
pred_list[:20]
sample= pd.read_csv('submission.csv')
len(test_D)
sample
result = {}
result['Id'] = list(i for i in range(1,10000))
result['Category'] = pred_list
pred_list.detach().numpy()
sample['Category'][1:] = pred_list.detach().numpy()
pd.DataFrame(sample)
sample.to_csv('q1.csv', index= False)