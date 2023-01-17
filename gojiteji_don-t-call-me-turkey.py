import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import torch

import torch.nn as nn

from torch.nn.utils import weight_norm

import torch.optim as optimizers

from tqdm import tqdm



#####データ前処理    

train = pd.read_json('train.json')

test = pd.read_json('test.json')

#train data

train_x_raw=np.array(train.audio_embedding)#[1195,10,128]たまに足りなくておかしい

train_x =np.zeros((1195,10,128))

##整形

for i in range(len(train_x_raw)):

    if(np.array(train_x_raw[i]).shape[0] is not 10):#足りない場合埋める

        lack=10 - np.array(train_x_raw[i]).shape[0]

        train_x[i]=np.append(train_x_raw[i],np.zeros((lack,128)),axis=0)

    else:#足りてたらnumpyに変換

        train_x[i]=np.array(train_x_raw[i])



train_y=train.is_turkey



#test data

test_x_raw=test.audio_embedding#[1196,10,128]

test_x =np.zeros((1196,10,128))

for i in range(len(test_x_raw)):

    if(np.array(test_x_raw[i]).shape[0] is not 10):#足りない場合埋める

        lack=10 - np.array(test_x_raw[i]).shape[0]

        test_x[i]=np.append(test_x_raw[i],np.zeros((lack,128)),axis=0)

    else:#足りてたらnumpyに変換

        test_x[i]=np.array(test_x_raw[i])



test_y=test.end_time_seconds_youtube_clip#回答を入れる配列





#####モデル本体

class ML(nn.Module):

    def __init__(self, device='cuda'):

        super().__init__()

        self.device = device

        self.conv1 = nn.Conv2d(1, 16,(2,2))

        self.conv2 = nn.Conv2d(16, 16, (2,2))

        self.conv3 = nn.Conv2d(16, 16, (2,2))

        self.pool2 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(p=0.4)

        #↓実装に応じていじる

        self.f0=nn.LeakyReLU()

        self.f1=nn.LeakyReLU()

        self.f2=nn.LeakyReLU()

        self.f3=nn.LeakyReLU()

        self.f4=nn.Sigmoid()

        self.l1 = nn.Linear(6240,25000)

        self.l  = nn.Linear(25000,1)

        self.pad = nn.ZeroPad2d(1)

    def forward(self,x):

        h=self.pad(x)

        h=self.conv1(h)

        h=self.f0(h)

        h=self.dropout(h)

        h=self.pad(h)

        h=self.conv2(h)

        h=self.f2(h)

        h=self.dropout(h)

        h=self.pad(h)

        h=self.conv3(h)

        h=self.f3(h)

        h=self.dropout(h)

        h=self.pool2(h)

        h=h.reshape(6240)

        h=self.f1(h)

        h=self.l1(h)

        h=self.l(h)

        h=self.dropout(h)

        return h

    

    

#####学習開始

device = torch.device('cuda')

model = ML(device=device).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optimizers.Adam(model.parameters())

def train_step(x,t):

    model.train()

    y=model(x)

    loss = criterion(y,t)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss

epochs =100

lists=np.array(range(len(train_x)))

histories=np.array([])

for epoch in tqdm(range(epochs)):

    train_loss = 0.

    np.random.shuffle(lists)

    for i in lists:

        x_tmp = np.array(train_x[i]).reshape(1,1,10,128)

        x = torch.from_numpy(x_tmp).type('torch.FloatTensor').to('cuda')#入力形式に変換(入力)

        t = torch.from_numpy(np.array([train_y[i]])).type('torch.FloatTensor').to('cuda')#入力形式に変換(出力)

        loss = train_step(x,t)#順伝播,逆伝播,更新

        train_loss += loss.item()

    train_loss /= len(train_x)

    histories=np.append(histories,train_loss)

    print('Epoch: {}, Cost: {:.3f}'.format(epoch+1,train_loss))



#####結果表示

plt.plot(histories)

plt.show()



#####自己評価

model.eval()

tmp=np.array([0]*len(train_x))

for i in range(len(train_x)):

    a=float(train_y[i])

    b=model(torch.from_numpy(np.array(train_x[i]).reshape(1,1,10,128)).type('torch.FloatTensor').to('cuda'))

    #print(b)

    b=torch.sigmoid(b)

    if(b>0.5):

        b=1

    else:

        b=0

    tmp[i]=abs(a-b)

print("acc:",100*(1-sum(tmp)/len(train_y)),"%")



#####test予測

for i in range(len(test_x)):

    b=model(torch.from_numpy(np.array(test_x[i]).reshape(1,1,10,128)).type('torch.FloatTensor').to('cuda'))

    b=torch.sigmoid(b)

    if(b>0.5):

        test_y[i]=1

    else:

        test_y[i]=0

submission = pd.read_csv('sample_submission.csv')

submission['is_turkey'] = np.array(test_y)

submission.to_csv('out.csv',index=False)