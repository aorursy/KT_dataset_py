!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c 2020-ai-exam-fashionmnist-2
!unzip 2020-ai-exam-fashionmnist-2
import pandas as pd
import numpy as np
import torch
train_csv = pd.read_csv('mnist_train_label.csv')
test_csv = pd.read_csv('mnist_test.csv') 
train_csv.head()
len(test_csv)
train_csv.describe()
train_L = train_csv.iloc[:,0]
train_D = train_csv.drop('9',axis =1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_D = scaler.fit_transform(train_D)# standardscaler 사용해서 정규화. 나머지는 1번과 동일. 
len(train_D)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
device= 'cpu'
l1 = torch.nn.Linear(train_D.shape[1], 10, bias = True)
torch.nn.init.xavier_uniform_(l1.weight)
#1번과 동일. 가중치 최적화 
model = torch.nn.Sequential(l1).to(device)
lr = 0.1
epoch = 15
batch_size= 1000
loss = torch.nn.CrossEntropyLoss().to(device) # to gpu
optimizer = torch.optim.SGD(model.parameters(), lr=lr) # 0.49
train_D = torch.FloatTensor(np.array(train_D))
train_L = torch.LongTensor(np.array(train_L))
#!nvidia-smi
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
sample.to_csv('q2.csv', index= False)