# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/titanic/train.csv')
train_df.head()
train_df = train_df.drop(['Ticket','Cabin'],axis=1)
train_df['Name']=train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Age'] = train_df['Age'].fillna(np.mean(train_df['Age']))
train_df.loc[train_df['Age']<=16,'Age']=0
train_df.loc[(train_df['Age']>16) & (train_df['Age']<=32),'Age']=1
train_df.loc[(train_df['Age']>32) & (train_df['Age']<=48),'Age']=2
train_df.loc[(train_df['Age']>48) & (train_df['Age']<=64),'Age']=3
train_df.loc[train_df['Age']>64,'Age']=4

train_df['Age'].value_counts()
train_df['Name']=train_df['Name'].replace(['Dr','Rev','Major','Col','Countess','Lady','Sir','Don','Jonkheer','Capt'],'rare')

train_df['Name']=train_df['Name'].replace(['Mlle','Ms'],'Miss')

train_df['Name']=train_df['Name'].replace('Mme', 'Mrs')

train_df['Embarked'] = train_df['Embarked'].fillna('S')
cat_col = ['Name','Sex','Embarked']
for cat in cat_col:
    train_df[cat]= train_df[cat].astype('category')
train_df.info()
cat_df = np.stack([train_df[cats].cat.codes.values for cats in cat_col],axis=1)
cat_df
train_df.columns
cont_col = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
cont_df = np.stack([train_df[con].values for con in cont_col],1)
cont_df
cont_df.dtype
import torch
import torch.nn as nn
cont_df = torch.tensor(cont_df,dtype=torch.float)
cont_df
cont_df
cat_df = torch.tensor(cat_df,dtype=torch.int64)
cat_df
Survived = train_df['Survived'].values
Survived = torch.tensor(Survived).flatten()
cat_szs = [len(train_df[col].cat.categories) for col in cat_col]

emb_szs = [(size,min(10,(size+1)//2)) for size in cat_szs]
emb_szs
class TabularModel(nn.Module):
    
    
    def __init__(self,emb_szs,n_cont,out_sz,layers,p=0.4):
        
        
        super().__init__()
        self.embds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(nf for ni,nf in emb_szs)
        n_in = n_emb + n_cont
        layerlist = []
        
            
        for i in layers:
                layerlist.append(nn.Linear(n_in,i))   
                layerlist.append(nn.ReLU(inplace=True))
                layerlist.append(nn.BatchNorm1d(i))
                layerlist.append(nn.Dropout(p))
                n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        self.layer = nn.Sequential(*layerlist)
        
        
    def forward(self,x_cat,x_cont):
        
            embeddings =[]
            
            for i,e in enumerate(self.embds):
                
                embeddings.append(e(x_cat[:,i]))
            x = torch.cat(embeddings,1)
            x = self.emb_drop(x)
            
            x_cont = self.bn_cont(x_cont)
            
            x = torch.cat([x,x_cont],1)
            
            x = self.layer(x)
            return x
torch.manual_seed(54)
model = TabularModel(emb_szs,cont_df.shape[1] , 2 ,[8,3],p=0.4)
model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 2500
losses =[]
for i in range(epochs):
    
    
    i += 1
    
    y_pred = model(cat_df,cont_df)
    loss = criterion(y_pred,Survived)
    losses.append(loss)
    if  i%100==0 :
        
        print("Epoch :",i,"loss: ",loss)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
test_df = pd.read_csv('../input/titanic/test.csv')
test_df
test_df = test_df.drop(['Ticket','Cabin'],axis=1)
test_df.columns
test_df.info()
train_df.info()
test_df['Age'] = test_df['Age'].fillna(np.mean(test_df['Age']))
test_df['Fare']= test_df['Fare'].fillna(np.mean(0))
test_df.loc[test_df['Age']<=16,'Age']=0
test_df.loc[(test_df['Age']>16) & (test_df['Age']<=32),'Age']=1
test_df.loc[(test_df['Age']>32) & (test_df['Age']<=48),'Age']=2
test_df.loc[(test_df['Age']>48) & (test_df['Age']<=64),'Age']=3
test_df.loc[test_df['Age']>64,'Age']=4

test_df['Age'].unique()
test_df['Name']=test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Name']=test_df['Name'].replace(['Dr','Rev','Major','Col','Countess','Lady','Sir','Don','Jonkheer','Capt','Dona'],'rare')

test_df['Name']=test_df['Name'].replace(['Mlle','Ms'],'Miss')

test_df['Name']=test_df['Name'].replace('Mme', 'Mrs')
train_df['Name'].unique()
test_df['Name'].unique()
Tcat_col = ['Name','Sex','Embarked']
for col in Tcat_col:
    test_df[col] = test_df[col].astype('category')
    
tcat_df = np.stack([test_df[col].cat.codes.values for col in Tcat_col],1)
import torch
import torch.nn as nn
tcat_df = torch.tensor(tcat_df,dtype=torch.int64)
tcat_df
tcont_col =    ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
A = test_df['PassengerId'].values
B =test_df['Pclass'].values
C = test_df['Age'].values
D = test_df['SibSp'].values
E = test_df['Parch'].values
F =test_df['Fare'].values

tcont_df = np.stack([test_df[col].values for col in tcont_col],1)
tcont_df
tcont_df = torch.tensor(tcont_df,dtype=torch.float)
tcont_df
tcat_df
with torch.no_grad():
    y_pred = model(tcat_df,tcont_df)
Survived = torch.argmax(y_pred,dim=1)
S = Survived.numpy()
S
subNN2 = pd.DataFrame({"PassengerId":test_df['PassengerId'],
                        "Survived" : S})
subNN2.to_csv('subNN2.csv')
