!pip uninstall -y kaggle

!pip install upgrade pip

!pip install kaggle==1.5.6



!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle

!chmod 600 ~/.kaggle/kaggle.json



import pandas as pd

import numpy as np
!kaggle competitions download -c sejongaiclasspredicteq



!unzip sejongaiclasspredicteq.zip
data=pd.read_excel('/content/train.xlsx')

data
# 데이터가 모두 한반도 상의 위치이므로, 위도 경도를 분류하는 부분을 제외하고, 진도만을 따로 데이터로 저장

data=np.array(data)

scale=data[:,2].reshape(-1,1)

scale
# Set Learning Parameter



learning_rate = 0.01

training_epochs = 2000

batch_size = 10
cnt=[]

for i in range(26):

  cnt.append([(i+25),0])



for i in range(26):

  k=scale>=(2.5+i*0.1)

  cnt[i][1]=k.sum()



cnt
x=[]

y=[]

for i in range(26):

  x.append(2.5+0.1*i)

  y.append(cnt[i][1])

X=np.array(x)

Y=np.array(y)
x
y
linear1 = torch.nn.Linear(26,256,bias=True)

linear2 = torch.nn.Linear(256,256,bias=True)

linear3 = torch.nn.Linear(256,256,bias=True)

linear4 = torch.nn.Linear(256,256,bias=True)

linear5 = torch.nn.Linear(256,1,bias=True)



# relu, dropout 선언

relu = torch.nn.ReLU()

dropout = torch.nn.Dropout(p=0.3)
# Random Init => Xavier Init

torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(linear4.weight)

torch.nn.init.xavier_uniform_(linear5.weight)
# relu는 마지막 레이어에서 뺄 것

# 레이어 연결



model = torch.nn.Sequential(linear1, relu, dropout,

                            linear2, relu, dropout,

                            linear3, relu, dropout,

                            linear4, relu, dropout,

                            linear5)
import torch

import torch.nn.functional as F

import torch.optim as optim

X=torch.FloatTensor(X)

Y=torch.FloatTensor(Y)
nb_epochs=5000

a=torch.ones(1,requires_grad=True)

b=torch.ones(1,requires_grad=True)

optimizer = torch.optim.Adam([a,b], lr=learning_rate)

for epoch in range(nb_epochs):

  h=model(10**(a-b*X))

  

  

  cost=torch.mean((h-Y)**2)



  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch%100==0:

    print(epoch ,cost)
Y
h
print(a,b)
ans=[]

for i in range(20):

  ans.append([i,0])

t=[2.0,2.1,2.2,2.3,2.4]

t=np.array(t)

t=torch.FloatTensor(t)



ans=10**(a-b*t)



ans
ar=[]

for i in range(5):

  ar.append(int((ans[i])*100))

ar=np.array(ar)



ar
truth=pd.read_csv('/content/samplesub.csv')

truth
sol=pd.DataFrame(ar)

sol=pd.DataFrame({"id":[0,1,2,3,4],"expected":ar})

sol
sol.to_csv('submit.csv',index=False,header=True)



!kaggle competitions submit -c sejongaiclasspredicteq -f submit.csv -m "submit"