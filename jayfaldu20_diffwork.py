# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

# Any results you write to the current directory are saved as output.
 !pip install fastai==0.7.0
from fastai.structured import * 


from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0']
df_test.head()
cat=['V2','V3','V4','V5','V7','V8','V9','V11','V16']

cont=['V1','V6','V10','V12','V13','V14','V15']

pred='Class'
df_train.head()
df_test[cont].describe()
df_train[cont].describe()


df_train.head()
df_train[cat].describe()
df_test[cat].describe()



train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']
df_test = df_test.loc[:, 'V1':'V16']

# df_test=df_test.drop(['V14'],axis=1)
for i in cat:

    print(i,np.sort(df_test[i].unique()))
for i in cat:

    print(i,np.sort(train_X[i].unique()))
for i in cat:

    if len(train_X[i].unique())>2:

        print(i)

        for j in np.sort(train_X[i].unique()):

            train_X[str(i)+'_'+str(j)]=(train_X[i]==j)

        train_X=train_X.drop([i],axis=1)
train_X.columns
for i in cat:

    if len(df_test[i].unique())>2:

        print(i)

        for j in np.sort(df_test[i].unique()):

            df_test[str(i)+'_'+str(j)]=(df_test[i]==j)

        df_test=df_test.drop([i],axis=1)

#         print(df_test.columns)
df_test.columns
df_test=df_test.drop(['V11_7'],axis=1)

df_test=df_test.drop(['V11_0'],axis=1)

df_test=df_test.drop(['V11_11'],axis=1)

# df_test=df_test.drop(['V14'],axis=1)
# train_X=train_X.drop(['V14'],axis=1)
df_test.columns,train_X.columns
mean={}

std={}

for j in cont:

#     print(train_X[j].mean())

    mean[j]=train_X[j].mean()

    std[j]=train_X[j].std()

    train_X[j]=(train_X[j]-mean[j])/std[j]

    df_test[j]=(df_test[j]-mean[j])/std[j]
len(train_X)
import torch
zz=torch.zeros(len(train_X.columns),len(train_X))

for idx,i in enumerate(train_X.columns):

    a=train_X[i].values

    zz[idx]=torch.tensor(a)
test_torch=torch.zeros(len(df_test.columns),len(df_test))

for idx,i in enumerate(df_test.columns):

    test_torch[idx]=torch.tensor(df_test[i].values)
train_X.describe()
df_test.describe()
train_X.describe()
df_test.describe()
train = train_X[:20000]

test = train_X[20000:]

train_y_train=train_y[:20000]

train_y_val=train_y[20000:]

# train=train.drop(['V14'],axis=1)

# test=test.drop(['V14'],axis=1)

def print_score(rf):

    print(rf.score(train,train_y_train),roc_auc_score(train_y_train,rf.predict_proba(train)[:,1]))

    print(rf.score(test,train_y_val),roc_auc_score(train_y_val,rf.predict_proba(test)[:,1]))

    
# rf = RandomForestClassifier(n_estimators=80,min_samples_leaf=5)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=5)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=5,max_features='log2')

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features='log2')

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=60,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=80,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=100,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=120,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=150,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=3,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=120,min_samples_leaf=3,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_features=0.5,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=3,max_features=0.5,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=110,min_samples_leaf=3,max_features=0.5,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=120,min_samples_leaf=3,max_features=0.5,random_state=123)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=110,min_samples_leaf=3,max_features=0.5,random_state=123,bootstrap=True)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=110,min_samples_leaf=3,max_features=0.5,random_state=123,bootstrap=False)

# rf.fit(train,train_y_train)

# print_score(rf)
# rf = RandomForestClassifier(n_estimators=120,min_samples_leaf=3,max_features=0.5,random_state=123,bootstrap=False)

# rf.fit(train,train_y_train)

# print_score(rf)
# best model found by hyperparameter tuning

rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_features=0.5,random_state=123)

rf.fit(train,train_y_train)

print_score(rf)
# doing feature importance using fastai's function

fd=rf_feat_importance(rf,train_X)

fd
fd[33:]
train_X=train_X.drop(fd.cols[33:],axis=1)
df_test=df_test.drop(fd.cols[33:],axis=1)
rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=3,random_state=123)

rf.fit(train_X,train_y)

rf.score(train_X,train_y)
pred=rf.predict_proba(df_test)
pred[:10]
cnt=0;

for i in range(pred.shape[0]):

    if pred[i,1]>=0.5: cnt+=1

cnt
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)
train_X.columns,df_test.columns
# also tried regressor which gave the same ans as classifier

# m=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=123)

# m.fit(train_X,train_y)

# m.score(train_X,train_y)
import torch.nn as nn
def kaim_uniform(m):

    if isinstance(m, nn.Linear):

        torch.nn.init.kaiming_normal(m.weight.data)

        

# using softmax in the outmost layer as we have to predict the probability        

def get_model():

  model=nn.Sequential(nn.Linear(45,100),

                      nn.BatchNorm1d(100),

                      nn.ReLU(),

                      nn.Linear(100,200),

                      nn.BatchNorm1d(200),

                      nn.ReLU(),

                      nn.Linear(200,100),

                      nn.BatchNorm1d(100),

                      nn.ReLU(),

                      nn.Linear(100,50),

                      nn.BatchNorm1d(50),

                      nn.ReLU(),

                      nn.Linear(50,25),

                      nn.BatchNorm1d(25),

                      nn.ReLU(),

                      nn.Linear(25,10),

                      nn.BatchNorm1d(10),

                      nn.ReLU(),

                      nn.Linear(10,2),

                      nn.Softmax()

                     )

  model.apply(kaim_uniform)

  return model
model=get_model()
import torch.nn.functional as F

def accuracy(out,yb): return (torch.argmax(out,dim=1)==yb).float().mean()

# Tried NLLLoss as it goes well with softmax and RMSE loss as we have to minimize the margin of probability

def RMSELoss(yhat,y):

    return torch.sqrt(torch.mean((yhat[:,1]-y.float()))**2)

loss_func= nn.NLLLoss()
n=len(train_y)

bs=64

x_tmp=zz[:,:64]

x_tmp=x_tmp.permute(1,0)

print(x_tmp.shape)

pred=model(x_tmp)

pred.shape,pred[0]
yy=np.asarray(df_train['Class'])

yy=torch.tensor(yy)
zz.shape[1]
X_train=zz[:,:20000]

y_train=yy[:20000]

# y_train=torch.tensor(y_train)

X_val=zz[:,20000:]

y_val=yy[20000:]

# y_val=torch.tensor(y_val)
X_train.shape,y_train.shape
y_val.shape
def get_batches(y_trian,bs):

  for n in range(0,len(y_train),bs):

    yield X_train[:,n:n+bs].permute(1,0),y_train[n:n+bs]
X_train.shape,y_train.shape
from torch import optim

import matplotlib.pyplot as plt
model=get_model()
bs=128

def train_loop(epoch,lr):

  train_loss=[]

  val_loss=[]

  acc=[]

  opt=optim.Adam(model.parameters(),lr)

  for epoch in range(epoch):

    lol=0

    cnt=0

    batch=get_batches(y_train,bs)

    for xb,yb in batch:

      cnt+=1;

#       print(cnt)

      pred=model(xb)

      loss=loss_func(pred, yb)

      lol+=loss

      loss.backward()

      opt.step()

      opt.zero_grad()

    ans=model(X_val.permute(1,0))

    val_tmp=loss_func(ans,y_val)

    val_loss.append(val_tmp)

    print("validation-{}".format(val_loss[epoch]))

    acc.append(accuracy(ans,y_val))

    print("accuracy-{}".format(acc[-1]))

    train_loss.append(lol/cnt)

    print("train_loss-{}".format(train_loss[-1]))

  plt.plot(val_loss,label='val_loss')

  plt.plot(train_loss,label='train_loss')

  plt.plot(acc,label='acc')

  plt.legend()
train_loop(2,0.003)
train_loop(2,0.0001)
train_loop(10,0.00002)
pre=model(X_val.permute(1,0))
pre=pre.squeeze()
pre[:10,:]
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)
test_index=df_test['Unnamed: 0']

train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']
df_test = df_test.loc[:, 'V1':'V16']

cat=['V2','V3','V4','V7','V8','V9','V11','V14','V16']

cont=['V1','V6','V10','V12','V13','V15']

pred='Class'
mean={}

std={}

for j in cont:

#     print(train_X[j].mean())

    mean[j]=train_X[j].mean()

    std[j]=train_X[j].std()

    train_X[j]=(train_X[j]-mean[j])/std[j]

    df_test[j]=(df_test[j]-mean[j])/std[j]
cor= df_train[cont].corr(method='pearson')

print(cor)
fig, ax =plt.subplots(figsize=(8, 6))

plt.title("Correlation Plot")

sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.show()
for i in cat:

    csq=chi2_contingency(pd.crosstab(train_y, train_X[i]))

    print("P-value of {}:{} ".format(i,csq[1]))
pd.crosstab(train_y, train_X[i])
for i in cat:

    fig, ax=plt.subplots(figsize=(8,6))

    sns.countplot(x=train_y, data=train_X, hue=i)

    plt.show()
for i in cat:

    fig, ax=plt.subplots(figsize=(8,6))

    sns.countplot(x=[1 for i in range(len(df_test))],data=df_test, hue=i)

    plt.show()
df_test=df_test.drop(['V11_7'],axis=1)

df_test=df_test.drop(['V11_0'],axis=1)

df_test=df_test.drop(['V11_11'],axis=1)
# df_test=df_test.drop(['V16'],axis=1)

df_test=df_test.drop(['V5'],axis=1)

# df_test=df_test.drop(['V11'],axis=1)

# train_X=train_X.drop(['V11'],axis=1)

train_X=train_X.drop(['V5'],axis=1)

for i in cat:

    print(i)

    for j in np.sort(train_X[i].unique()):

      train_X[str(i)+'_'+str(j)]=(train_X[i]==j)

    train_X=train_X.drop([i],axis=1)

for i in cat:

    print(i)

    for j in np.sort(df_test[i].unique()): 

        df_test[str(i)+'_'+str(j)]=(df_test[i]==j)

    df_test=df_test.drop([i],axis=1)

df_test=df_test.drop(['V14'],axis=1)

train_X=train_X.drop(['V14'],axis=1)



# remove V14 as their is large difference between the mean of test and train data
rf=RandomForestClassifier(random_state=42)
param_grid = { 

    'n_estimators': [50, 200,100],

    'max_features': ['auto', 'sqrt', 'log2',0.5],

    'max_depth':[4,6,8,16,20],

    'criterion' :['gini', 'entropy']

}
CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

CV_rf.fit(train_X, train_y)

CV_rf.best_params_
X_train=train_X[:20000]

X_test=train_X[20000:]

train_y_train=train_y[:20000]

train_y_test=train_y[20000:]
def print_score(m):

    print(m.score(X_train,train_y_train),roc_auc_score(train_y_train,m.predict_proba(X_train)[:,1]))

    print(m.score(X_test,train_y_test),roc_auc_score(train_y_test,m.predict_proba(X_test)[:,1]))
#This didn't worked out very well

m=RandomForestClassifier(criterion='gini',max_depth=6,max_features=0.5,n_estimators=100)

m.fit(X_train,train_y_train)

print_score(m)
# after much tuning hyperparameters I found this one best on validation set

m=RandomForestClassifier(criterion='entropy',max_depth=8,n_estimators=300,random_state=42,max_features=0.5)

m.fit(X_train,train_y_train)

print_score(m)
m=RandomForestClassifier(criterion='entropy',max_depth=8,max_features=0.5,n_estimators=200,random_state=42)

m.fit(train_X,train_y)

print_score(m)
fi=rf_feat_importance(m,train_X)

fi
pred=m.predict_proba(df_test)

pred
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('final_2.csv', index=False)