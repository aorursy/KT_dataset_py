# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/online-retail-data-set-from-ml-repository/retail_dataset.csv')
df.head()
df.fillna('<nothing>',inplace=True)
labels=df.values.reshape(-1)
labels=set(labels)
labels        
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ohe.fit(np.asarray(list(labels)).reshape(-1,1))
ohe.categories_
ohe.transform([[df['0'][4]]]).toarray()
import torch as T
import torch.nn as nn
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Network,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True) 
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU())
        self.out=nn.Sequential(
                    nn.Linear(hidden_dim,output_size),
                    nn.LogSoftmax()
        
        )
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out=self.out(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = T.zeros(self.n_layers, batch_size, self.hidden_dim)
    
        return hidden 
model=Network(input_size=len(labels),output_size=len(labels),hidden_dim=8,n_layers=1)
model
model.forward(T.from_numpy(ohe.transform([[df['0'][0]]]).toarray()).unsqueeze(0).float())
X=np.zeros((1,7,10))

for x in range(len(df)):
    X=np.concatenate((X,ohe.transform(df.values[x].reshape(-1, 1)).toarray().reshape(1,7,10)),axis=0)
X.shape
model.forward(T.from_numpy(X[1]).float().unsqueeze(0))
X=T.from_numpy(X[1:]).float()
X
optimizer=T.optim.Adam(model.parameters())
creterion = nn.CrossEntropyLoss()
out=model.forward(X[0].unsqueeze(0))
out
Y=X[:,1:,:].max(axis=2)[1]
X=X[:,0:-1,:]
out=model.forward(X[0].unsqueeze(0))
X[:,0:,:].shape
creterion(out[0],Y[0])
X.shape

out=model.forward(X)
creterion(out[0],Y.view(-1,1).squeeze(1))
#normalizing
s=df['0'].value_counts()+df['1'].value_counts()+df['3'].value_counts()+df['2'].value_counts()+df['4'].value_counts()+df['5'].value_counts()+df['6'].value_counts()
s=2*s
s[0]=50
s
s=6*s/sum(s)
s
mask=T.from_numpy(s.to_numpy())
mask
X=X*mask
losses=[]
for i in range(6500):
        
   optimizer.zero_grad()
   out=model.forward(X.float())
   loss=creterion(out[0],Y.view(-1,1).squeeze(1))
        
   losses.append(loss.item())
   loss.backward()
   if(i%500==0):
        print('loss is {}'.format(loss.item()))
   optimizer.step()
from matplotlib import pyplot as plt
plt.plot(losses)
def predict(model,item):
#     item=ohe.transform(item)
    out, hidden = model(item)
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    idx = T.topk(prob,k=4, dim=0)
    return idx, hidden
p=predict(model,X[2,0].reshape([1,1,-1]).float())
p[0][1]
def conv_from_idx(idx):
    return ohe.categories_[0][idx]
conv_from_idx(p[0][1][0])
from random import randint
def sample(model,inital_data):
    l=[]
    for i in inital_data:

        p=predict(model,i.float())
        print(conv_from_idx(p[0][1][0]))
        l.append(p[0])
    return l
X.shape
for i in range(100):
    l = []
    for j in range(3):
        l.append(conv_from_idx(X[i][j].max(axis=0)[1]))
        print('for {} consumer can buy '.format(l),end=' ')
        sample(model,[X[i][0:j+1].unsqueeze(0).float()])
