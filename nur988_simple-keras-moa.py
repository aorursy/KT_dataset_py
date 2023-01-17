import pandas as pd

import random

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import os

import tensorflow as tf

import copy

import seaborn as sns

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
train_features=pd.read_csv("../input/lish-moa/train_features.csv")

train_targets_scored=pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_targets_nonscored=pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

test_features=pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission=pd.read_csv('../input/lish-moa/sample_submission.csv')
GENES=[col for col in train_features.columns if col.startswith('g-')]

CELLS=[col for col in train_features.columns if col.startswith('c-')]
train_features.describe()
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=42)
n_comp=30

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train = pd.concat((train_features[['cp_type','cp_time','cp_dose']], train2), axis=1)

test = pd.concat((test_features[['cp_type','cp_time','cp_dose']], test2), axis=1)
train_features.shape
test_features.isnull().sum()
#CELLS

n_comp = 15



data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train = pd.concat((train, train2), axis=1)

test = pd.concat((test, test2), axis=1)
#train=train_features

#test=test_features

from sklearn.feature_selection import VarianceThreshold





var_thresh=VarianceThreshold(threshold=0.5)
data=train.append(test)

data.shape
data_transformed=var_thresh.fit_transform(data.iloc[:,4:])

data_transformed.shape
data.head()
train_transformed=data_transformed[:train_features.shape[0]]

test_transformed=data_transformed[-test_features.shape[0]:]
train_transformed.shape,test_transformed.shape
train_features=pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1,4),columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features=pd.concat([train_features,pd.DataFrame(train_transformed[:,4:])],axis=1)

 

train_features.shape
test_features=pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1,4),columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features=pd.concat([test_features,pd.DataFrame(test_transformed[:,4:])],axis=1)
train=train_features.merge(train_targets_scored,on='sig_id')

target=train[train_targets_scored.columns]
target.shape,train_features.shape

#train=train.drop('cp_type',axis=1)

#test=test_features.drop('cp_type',axis=1)

train.isnull()
train=train.loc[train['cp_type']!='ctl_vehicle']
train=train_features.drop('sig_id',axis=1)

test=test_features.drop('sig_id',axis=1)
COLS=['cp_type','cp_time','cp_dose']

FE=[]

for col in COLS:

    for mod in train[col].unique():

        FE.append(mod)

        train[mod]=(train[col]==mod).astype(int)
COLS=['cp_type','cp_time','cp_dose']

FE=[]

for col in COLS:

    for mod in test[col].unique():

        FE.append(mod)

        test[mod]=(test[col]==mod).astype(int)
train=train.drop(['cp_type','cp_time','cp_dose'],axis=1)

test=test.drop(['cp_type','cp_time','cp_dose'],axis=1)
target_cols=target.drop('sig_id',axis=1).columns.values.tolist()

target=target.drop('sig_id',axis=1)
target_cols
#import xgboost as xgb

#from sklearn.model_selection import train_test_split

#from sklearn.multioutput import MultiOutputClassifier

#from sklearn.metrics import accuracy_score



#x_train,x_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=42)

#xgb_estimator=xgb.XGBClassifier(objective='binary:logistic')

#multilabel_model=MultiOutputClassifier(xgb_estimator)

#multilabel_model.fit(x_train,y_train)
train=train.astype(float)
target=target.astype(float)
train.isnull().sum()
num_columns=train.shape[1]
from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,BatchNormalization

import keras

from keras.optimizers import SGD

import tensorflow as tf

train.shape,test.shape
model=Sequential()

model.add(BatchNormalization())

model.add(Dropout(0.20))

model.add(Dense(2048,activation='relu',input_dim=train.shape[1],kernel_initializer='uniform'))

model.add(BatchNormalization())

model.add(Dense(2048,activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(1048,activation='tanh'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

#model.add(Dense(1024,kernel_initializer='uniform',activation='tanh'))

#model.add(Dropout(0.5))

model.add(Dense(target.shape[1],activation='sigmoid'))

sgd=SGD(lr=0.01,momentum=0.9)

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train.values,target.values,batch_size=60,epochs=40,verbose=2)
#preds=multilabel_model.predict(x_test)

#accuracy_score(y_test,preds)*100
preds=model.predict(test)
id=test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']
preds[0]
df=pd.DataFrame(preds,columns=list(target),index=test_features['sig_id'])

df.index[0]
for i in range(len(df.index)):

    if df.index[i] in(id):

        df.iloc[df.index[i],train_targets_scored.columns[1:]]=0
df.to_csv('submission.csv')