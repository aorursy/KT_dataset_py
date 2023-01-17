! pip install skorch==0.8.0
! pip install lofo-importance
import pandas as pd
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance, FLOFOImportance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import log_loss
import numpy as np
%matplotlib inline
train = pd.read_csv('../input/lish-moa/test_features.csv')
train = train[train.cp_type!='ctl_vehicle']
train_labels = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
gene_cols = [col for col in train.columns if 'g-' in col]
labels = train_labels.columns.difference(['sig_id']).tolist()
train_df = train.drop(['sig_id'], axis=1)[gene_cols+['cp_time']]
label_df = train_labels[labels]
df = pd.concat([train_df,label_df],axis=1).dropna().reset_index(drop=True)
l = 79
label_df.iloc[:,l].value_counts()
sss = StratifiedShuffleSplit(n_splits=2,test_size=0.28, random_state=42)
sp1,sp2 = sss.split(df[[labels[l]]+['cp_time']].values,df[[labels[l]]+['cp_time']].values)
train_df = df.iloc[list(sp1[0]),:].reset_index(drop=True)
val_df = df.iloc[list(sp1[1]),:].reset_index(drop=True)
rf = RandomForestClassifier(n_jobs=-1, max_depth=6)
rf.fit(train_df[gene_cols],train_df[labels[l]])
lofo_imp = FLOFOImportance(trained_model=rf, validation_df=val_df,features=gene_cols,target=labels[l],scoring='neg_log_loss',n_jobs=-1)
imps = lofo_imp.get_importance()
plot_importance(imps[:40])
plot_importance(imps[-40:])
label = label_df.columns[l]
X,y = df[gene_cols], df[label]
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1881).split(y,y)
scores = []

for i, (tr_ix, vl_ix) in enumerate(skf):
    print(f'================Fold-{i+1}===================')
    
    train_X, train_y = X.iloc[tr_ix], y.iloc[tr_ix]
    valid_X, valid_y = X.iloc[vl_ix], y.iloc[vl_ix]
    
    model = RandomForestClassifier(max_depth=6, n_jobs=-1, n_estimators=300)
    
    model.fit(train_X, train_y)
    
    preds = model.predict_proba(valid_X)[:,1]
    
    score = log_loss(valid_y, preds)
    
    scores.append(score)
    
print(np.mean(scores))
elim = 40
reduced_feats = imps.feature.values[:elim].tolist()
X,y = df[reduced_feats], df[label]
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1881).split(y,y)
scores = []

for i, (tr_ix, vl_ix) in enumerate(skf):
    print(f'================Fold-{i+1}===================')
    
    train_X, train_y = X.iloc[tr_ix], y.iloc[tr_ix]
    valid_X, valid_y = X.iloc[vl_ix], y.iloc[vl_ix]
    
    model = RandomForestClassifier(max_depth=6, n_jobs=-1, n_estimators=300)
    
    model.fit(train_X, train_y)
    
    preds = model.predict_proba(valid_X)[:,1]
    
    score = log_loss(valid_y, preds)
    
    scores.append(score)
    
print(np.mean(scores))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import skorch
from skorch import NeuralNetBinaryClassifier
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        
        self.input = nn.Linear(self.input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.hidden1 = nn.Linear(512,256)
        self.hidden2 = nn.Linear(256,128)
        self.out = nn.Linear(128,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()
        
    def forward(self, X):
        
        x = self.input(X)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.out(x)
        out = self.sig(x)
        
        return out
        
        
X,y = df[gene_cols], df[label]
skorch.__version__
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1881)
model = NeuralNetBinaryClassifier(Net(X.shape[1]),
                                      optimizer=torch.optim.Adam,
                                      train_split=skorch.dataset.CVSplit(skf),
                                      max_epochs=10,
                                      criterion=torch.nn.modules.loss.BCELoss)
model.fit(torch.Tensor(X.values),torch.Tensor(y.values))
elim = 40
reduced_feats = imps.feature.values[:elim].tolist()
X,y = df[reduced_feats], df[label]
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1881)
model = NeuralNetBinaryClassifier(Net(X.shape[1]),
                                      optimizer=torch.optim.Adam,
                                      train_split=skorch.dataset.CVSplit(skf),
                                      max_epochs=10,
                                      criterion=torch.nn.modules.loss.BCELoss)
model.fit(torch.Tensor(X.values),torch.Tensor(y.values))
