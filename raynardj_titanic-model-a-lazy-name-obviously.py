# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/gender_submission.csv")
bs = 32
train_df.sample(100)
def df_rich(df):
    df["has_age"] = (~df.Age.isna())*1
    df["Age2"] = df.Age.fillna(30)
    df["hasParch"] = (df["Parch"]>0)*1
    df["hasSibSp"] = (df["SibSp"]>0)*1
    df["hasCabin"] = (~df.Cabin.isna())*1
    return df
train_df = df_rich(train_df)
test_df = df_rich(test_df)
train_df.head()
import matchbox,b4tab
Sex = b4tab.categorical_idx("Sex",)
Sex.build(train_df.Sex)
has_age = b4tab.categorical_idx("has_age")
has_age.build(train_df.has_age)
Embarked = b4tab.categorical_idx("Embarked")
Embarked.build(train_df.Embarked)
Pclass = b4tab.categorical_idx("Pclass")
Pclass.build(train_df.Pclass)
hasParch = b4tab.categorical_idx("hasParch")
hasParch.build(train_df.hasParch)
hasSibSp = b4tab.categorical_idx("hasSibSp")
hasSibSp.build(train_df.hasSibSp)
hasCabin = b4tab.categorical_idx("hasCabin")
hasCabin.build(train_df.hasCabin)

Age2 = b4tab.minmax("Age2")
Age2.build(train_df.Age2)
Fare = b4tab.minmax("Fare")
Fare.build(train_df.Fare)


Survived = b4tab.categorical_idx("Survived")
Survived.build(train_df.Survived)
x_input = b4tab.tabulate("x_input")
x_input.build(Sex,has_age,hasCabin,hasParch,hasSibSp,Pclass,Embarked,Age2,Fare)
y_train = Survived.prepro(train_df.Survived)
x_train = x_input.prepro(train_df)
x_test = x_input.prepro(test_df)
x_train.shape
emb_len = dict((k,len(v['idx2cate'])) for k,v in x_input.meta["cols"].items() if v["coltype"]=='categorical_idx')
emb_len
from torch import nn
import torch
class structured(nn.Module):
    def __init__(self,emb_len,input_width,hs):
        super(structured,self).__init__()
        self.hs = hs
        for k,v in emb_len.items():
            setattr(self,k,nn.Embedding(v,self.hs))
        self.emb_keys = list(emb_len.keys())
        self.width = len(self.emb_keys)*self.hs + (input_width - len(self.emb_keys))
        self.dnn = nn.Sequential(*[
            nn.Linear(self.width,128,bias=False),
            nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,1),
            nn.Sigmoid(),
        ])
    
    def forward(self,x):
        embx = torch.cat(list(getattr(self,self.emb_keys[e])(x[:,e].long()) for e in range(len(self.emb_keys))),dim = 1)
        x = torch.cat([embx,x[:,len(self.emb_keys):]],dim=1)
        return self.dnn(x)
    
md = structured(emb_len=emb_len,input_width=x_train.shape[1],hs=15)
from torch.optim import Adam
from torch.nn import BCELoss
from matchbox import Arr_Dataset

opt = Adam(md.parameters())
lossf = BCELoss()
train = Arr_Dataset(x_train, y_train,bs=bs)
valid = Arr_Dataset(x_test,bs=bs)
def action(*args,**kwargs):
    x,y = args[0]
    x,y = x.float().squeeze(0),y.float().squeeze(0)
    opt.zero_grad()
    
    y_ = md(x)
    loss=lossf(y_,y)
    loss.backward()
    opt.step()
    return {"loss":loss.item()}
t = matchbox.Trainer(train,batch_size=1,print_on=2,)
t.action = action
t.train(25)
result = (np.concatenate(list(md(torch.FloatTensor(x_test)[i*bs:(i+1)*bs,:]).data.numpy() for i in range(x_test.shape[0]//bs+1)),axis=0).reshape(-1)>0.5)*1    
test_df["Survived"] = result
test_df["Survived"] = test_df["Survived"].apply(lambda x:int(x))
test_df[["PassengerId","Survived"]].to_csv("submission.csv",index=False)