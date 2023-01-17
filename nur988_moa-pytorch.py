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

from sklearn.preprocessing import StandardScaler,QuantileTransformer

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
for col in (GENES+CELLS):

    transformer=QuantileTransformer(n_quantiles=100,random_state=0,output_distribution="normal")

    vec_len=len(train_features[col].values)

    vec_len_test=len(test_features[col].values)

    raw_vec=train_features[col].values.reshape(vec_len,1)

    transformer.fit(raw_vec)

    train_features[col]=transformer.transform(raw_vec).reshape(1,vec_len)[0]

    test_features[col]=transformer.transform(test_features[col].values.reshape(vec_len_test,1)).reshape(1,vec_len_test)[0]
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
train.shape,test.shape
target.head()
train.head()
class MoaDataset:

    def __init__(self,features,targets):

        self.features=features

        self.targets=targets

        

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self,item):

        return{

            'x':torch.tensor(self.features[item,:],dtype=torch.float),

            'y':torch.tensor(self.targets[item,:],dtype=torch.float)

        }

    

    

class TestDataset:

    def __init__(self,features):

        self.features=features

        

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self,item):

        return{

            "x":torch.tensor(self.features[item,:],dtype=torch.float)

        }

    
params= {'num_layer': 3, 'hidden_size': 1404, 'dropout': 0.10089577325447585, 'learning_rate': 3.764735971734488e-05}

#Best is trial 39 with value: 0.00175025769731.
class Model(nn.Module):

    def __init__(self,n_features,n_targets):

        super(Model,self).__init__()

        self.batch_norm=nn.BatchNorm1d(n_features)

        self.dropout1=nn.Dropout( 0.10089577325447585)

        self.dense1=nn.Linear(n_features,1404)

        

        self.batch_norm2=nn.BatchNorm1d(1404)

        self.dropout2=nn.Dropout( 0.10089577325447585)

        self.dense2=nn.Linear(1404,1404)

        

        self.batch_norm3=nn.BatchNorm1d(1404)

        self.dropout3=nn.Dropout( 0.10089577325447585)

        self.dense3=nn.Linear(1404,1404)

        

       

        self.batch_norm4=nn.BatchNorm1d(1404)

        self.dropout4=nn.Dropout( 0.10089577325447585)

        self.dense4=nn.Linear(1404,206)

        

    def forward(self,x):

        x=self.batch_norm(x)

        x=self.dropout1(x)

        x=F.relu(self.dense1(x))

        

        x=self.batch_norm2(x)

        x=self.dropout2(x)

        x=F.relu(self.dense2(x))

        

        #x=self.batch_norm3(x)

        #x=self.dropout3(x)

        #x=F.relu(self.dense3(x))

        

        x=self.batch_norm4(x)

        x=self.dropout4(x)

        x=self.dense4(x)

        

        return x

    

        
EPOCHS=30

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset=MoaDataset(train.values,target.values)

test_dataset=TestDataset(test.values)
trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)

testloader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False)

model=Model(n_features=train.shape[1],n_targets=target.shape[1])

model.to(DEVICE)

opt=torch.optim.Adam(model.parameters(),lr= 3.764735971734488e-05)

loss_fn=nn.BCEWithLogitsLoss()
model.train()



for epochs in range(EPOCHS):

    final_loss=0

    for data in trainloader:

        opt.zero_grad()

        inputs,targets=data['x'].to(DEVICE),data['y'].to(DEVICE)

        outputs=model(inputs)

        loss=loss_fn(outputs,targets)

        loss.backward()

        opt.step()

        final_loss+=loss.item()

    final_loss/=len(trainloader)  

    

    print(f"Epoch{epochs}--{final_loss}")
model.eval()

preds = []

    

for data in testloader:

    inputs = data['x'].to(DEVICE)



    with torch.no_grad():

        outputs = model(inputs)

        

    preds.append(outputs.sigmoid().detach().cpu().numpy())

        

preds = np.concatenate(preds)
id=test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']
preds[0]
df=pd.DataFrame(preds,columns=list(target),index=test_features['sig_id'])

df.head()
df.to_csv('submission.csv')