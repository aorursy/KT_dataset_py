import os
import numpy as np # linear algebra
import pandas as pd 
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SparseAdam,Adam,Adagrad,SGD
#from livelossplot import PlotLosses
COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
train_data = pd.read_csv("../input/ml-100k/u1.base",sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
test_data = pd.read_csv("../input/ml-100k/u1.test",sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
n_users, n_items = 943,1682
class Matrixfactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding( n_users,n_factors)
        self.item_factors = nn.Embedding( n_items,n_factors)
#         self.l1 = nn.Linear(n_factors*2,16)
#         self.l2 = nn.Linear(16,1)
#         self.drop = nn.Dropout()
        
    def forward(self, user, item):
        user = torch.LongTensor(user) - 1
        item = torch.LongTensor(item) - 1
        u,it = self.user_factors(user),self.item_factors(item)
#         x = torch.cat([u,it],dim=1)
#         x = F.relu(self.l1(x))
#         x = self.drop(x)
#         x = self.l2(x)
#         return x
        x = (u*it).sum(1)
        assert x.shape==user.shape
        return x * 5
model = Matrixfactorization(n_users,n_items)
opt = Adam(model.parameters(),lr=1e-3)
criterion = nn.L1Loss()
BATCH_SIZE = 32
avg = []
mx = []
states = {}
model.train(True)
for e in range(20):
    for it in range(len(train_data)//BATCH_SIZE):
        #---------------SETUP BATCH DATA-------------
        df = train_data.sample(frac=BATCH_SIZE/len(train_data))
        users = df.user_id.values
        items = df.movie_id.values
        targets = torch.FloatTensor(df.rating.values)
        assert users.shape==(BATCH_SIZE,)==items.shape
        
        #----------------TRAIN MODEL------------------------
        opt.zero_grad()
        preds = model(users,items)
        mx.append((preds.max().item(),preds.min().item()))
        loss = criterion(preds,targets)
        assert preds.shape==targets.shape
        loss.backward()
        opt.step()
        avg.append(loss.item())

#         if it%500==0:
#             print(f"Iter {it}: {sum(avg)/len(avg)}")

    print(f"EPOCH {e+1}:",sum(avg)/len(avg))
    avg = []
    states[e+1] = model.state_dict()

preds.view(-1).size()
with torch.no_grad():
    #model.load_state_dict(states[20])
    model.train(False)
    users = test_data.user_id.values
    items = test_data.movie_id.values
    test_data['pred'] = model(users,items).numpy()
mean_absolute_error(test_data.pred,test_data.rating)
test_data['pp'] = test_data.pred.clip(0,5).round()
print(test_data.rating.std(),test_data.pp.std(),test_data.pred.std())

ix = torch.LongTensor(list(range(n_users)))
a = model.user_factors(ix)
a.norm(dim=1).max(),a.norm(dim=1).std()
p = np.array(mx)
#test_data.pred.max(),test_data.pred.min(),
plt.plot(range(len(p)),p[:,0],'ro')  
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
plt.xlim(1,5)
sns.distplot(test_data.pred.values.clip(0,5),ax=ax1)
sns.distplot(test_data.rating.values,ax=ax2);

df.user_id.values

















