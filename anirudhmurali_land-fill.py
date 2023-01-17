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
data = pd.read_csv("../input/land-use-permits.csv")
data.head()
data.Status.value_counts()
data.shape
data.isnull().sum()
data.dropna(subset=["Status"],inplace=True) 
data.drop(["Contractor"],axis=1,inplace=True)
data.drop(["Application/Permit Number"],axis=1,inplace=True)
data.shape
data.Location[0]
data.drop(["Location"],axis=1,inplace=True)
data.shape
data["Permit and Complaint Status URL"][0]
data.drop(["Permit and Complaint Status URL"],axis=1,inplace=True)
data.shape
data.Address[:5]
len(data.Address.unique())
data.drop(["Address"],axis=1,inplace=True)
data.shape
data.drop(["Description"],axis=1,inplace=True)
data.shape
# data.Description[0]
len(data.Category.unique())
len(data["Decision Type"].unique())
zeros = data.loc[data.Value == 0]
zeros.shape
data.drop(["Value"],axis=1,inplace=True)
data.shape
data.drop(["Applicant Name","Application Date","Decision Date","Issue Date"],axis=1,inplace=True)
data.shape
data.drop(["Design Review Included"],axis=1,inplace=True)
data.shape
data.isnull().sum()
data.head()
permits = pd.DataFrame()
# permits["Application/Permit Number"] = data["Application/Permit Number"]
permits["Permit Type"] = data["Permit Type"]

permits["Permit Type"].fillna("EMPTY",inplace=True)
permits
# permits.dropna(subset=["Permit Type"],inplace=True)
import re
from progressbar import ProgressBar
permit_type =[]
pbar = ProgressBar()

for s in pbar(permits["Permit Type"]):
    parts = re.split(r'[.,]', s)
    for p in parts:
        if p[0] == ' ':
            permit_type.append(p[1:])
        else:
            permit_type.append(p)
ps = pd.DataFrame(permit_type,columns=["Permit"])
permit_type= ps.Permit.unique()
permit_type

# data["Permit Type"].fillna("  ",inplace=True)
for permit in permit_type:
    temp = []
    
    pbar = ProgressBar()
    for s in pbar(permits["Permit Type"]):
        parts = re.split(r'[.,]', s)
        flag =0
        for p in parts:
            if p[0] == ' ':
                p = p[1:]
#             else:
#                 permit_type.append(p)
            if p == permit:
                temp.append(1)
                flag = 1
        if flag ==0:
            temp.append(0)
    data[permit] = temp
data.drop(["EMPTY","Permit Type"],axis=1,inplace=True)
data.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

le =  LabelEncoder()
data.Category = le.fit_transform(data.Category.astype(str))
data["Decision Type"]= le.fit_transform(data["Decision Type"].astype(str))
data["Appealed?"]= le.fit_transform(data["Appealed?"].astype(str))
data["Status"]= le.fit_transform(data["Status"].astype(str))

kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
params = {
    'min_child_weight': 10.0,
    'objective': 'multi:softmax',
    'num_class':9,
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700,
    'eval':'mlogloss'
    }
# def gini(actual, pred, cmpcol = 0, sortcol = 1):
#     assert( len(actual) == len(pred) )
#     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
#     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
#     totalLosses = all[:,0].sum()
#     giniSum = all[:,0].cumsum().sum() / totalLosses
    
#     giniSum -= (len(actual) + 1) / 2.
#     return giniSum / len(actual)
 
# def gini_normalized(a, p):
#     return gini(a, p) / gini(a, a)

# def gini_xgb(preds, dtrain):
#     labels = dtrain.get_label()
#     gini_score = gini_normalized(labels, preds)
#     return 'gini', gini_score
data.dropna(subset=["Latitude","Longitude"],inplace=True)
X = data.drop(['Status'], axis=1).values
y = data.Status.values
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
#     d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist,early_stopping_rounds=70) #
    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, max_depth=7,random_state=0)
best = 0 
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    score = clf.score(X_valid,y_valid)
    if score > best:
        best = score
    print (score)
print("best  accuracy: "+ str(best) )
#     print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
best = 0 
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    lr.fit(X_train, y_train)
    score = lr.score(X_valid,y_valid)
    if score > best:
        best = score
    print (score)
print("best  accuracy: "+ str(best) )