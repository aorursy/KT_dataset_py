#kvsnoufal@gmail.com
#Version 2 of this notebook gave the best score
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

# Any results you write to the current directory are saved as output.
import pandas as pd
from scipy.stats import randint 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings("ignore")
def evaluate_mdl(X,y):
    
    scorer=make_scorer(log_loss, greater_is_better=False)

    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=0, cv=3,n_jobs=-1 ,scoring=scorer)
    grid_search_cv.fit(X, y)
    NFOLDS=5
    skf = StratifiedKFold(n_splits=NFOLDS)
    
    # preds=np.zeros((X_test.shape[0],))
    losses=[]
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # print(X_train.shape,X_val.shape)
        clf=grid_search_cv.best_estimator_
            
        clf.fit(X_train,y_train)
        pval=clf.predict_proba(X_val)[:,1]
        loss=log_loss(y_val,pval)
        losses.append(loss)
    print(np.mean(losses))
    return np.mean(losses)
# evaluate_mdl(X,y)
# test=pd.read_csv("/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv")
# test["target"]=-1


# data=pd.concat([train,test],ignore_index=True)
# data["occupation"].value_counts().to_dict()
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer,PowerTransformer,Normalizer,MaxAbsScaler,KBinsDiscretizer,RobustScaler
continuous=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
target=["target"]
id=["uid"]
categorical=['workclass','marital-status', 'occupation', 'relationship', 'race','sex', 'native-country','education']
train=pd.read_csv("../input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv")
test=pd.read_csv("../input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv")
test["target"]=-1


data=pd.concat([train,test],ignore_index=True)
data["workclass"]=data["workclass"].apply(str.strip).replace("?",np.nan)
data["workclass"].fillna(data["workclass"].mode()[0],inplace=True)

cnts=data[categorical]["native-country"].value_counts()
smallcnts=list(cnts[cnts<200].index)
data.loc[data["native-country"].isin(smallcnts),"native-country"]="others"




dummies=pd.get_dummies(data[categorical])
dcols=dummies.columns
print(data.shape)
data=data.merge(dummies,how='left',left_index=True,right_index=True)
for col in categorical:
    enc=LabelEncoder()
    data[col]=enc.fit_transform(data[col].values.reshape(-1,1)).flatten()

# data["net-cap"]=data["capital-gain"]-data["capital-loss"]

train=data[data["target"]!=-1]
test=data[data["target"]==-1]
X=train.drop(target+id+['workclass_Private', 'occupation_ Tech-support'],axis=1).values
y=train.target.values
"both dummy and label + scaling"
evaluate_mdl(X,y)
# 0.31566125458576005

from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer,PowerTransformer,Normalizer,MaxAbsScaler,KBinsDiscretizer,RobustScaler
continuous=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
target=["target"]
id=["uid"]
categorical=['workclass','marital-status', 'occupation', 'relationship', 'race','sex', 'native-country','education']
train=pd.read_csv("../input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv")
test=pd.read_csv("../input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv")
test["target"]=-1


data=pd.concat([train,test],ignore_index=True)
data["workclass"]=data["workclass"].apply(str.strip).replace("?",np.nan)
data["workclass"].fillna(data["workclass"].mode()[0],inplace=True)

cnts=data[categorical]["native-country"].value_counts()
smallcnts=list(cnts[cnts<100].index)
data.loc[data["native-country"].isin(smallcnts),"native-country"]="others"




dummies=pd.get_dummies(data[categorical])
dcols=dummies.columns
print(data.shape)
data=data.merge(dummies,how='left',left_index=True,right_index=True)
for col in categorical:
    enc=LabelEncoder()
    data[col]=enc.fit_transform(data[col].values.reshape(-1,1)).flatten()

# data["net-cap"]=data["capital-gain"]-data["capital-loss"]

train=data[data["target"]!=-1]
test=data[data["target"]==-1]
X=train[['age',
 'workclass',
 'fnlwgt',
 'education',
 'education-num',
 'marital-status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'capital-gain',
 'capital-loss',
 'hours-per-week',
 'native-country',
 'workclass_Federal-gov',
 'workclass_Local-gov',
 'workclass_Self-emp-inc',
 'workclass_Self-emp-not-inc',
 'marital-status_ Married-AF-spouse',
 'marital-status_ Married-civ-spouse',
 'marital-status_ Married-spouse-absent',
 'marital-status_ Never-married',
 'marital-status_ Separated',
 'marital-status_ Widowed',
 'occupation_ ?',
 'occupation_ Adm-clerical',
 'occupation_ Craft-repair',
 'occupation_ Exec-managerial',
 'occupation_ Farming-fishing',
 'occupation_ Handlers-cleaners',
 'occupation_ Machine-op-inspct',
 'occupation_ Other-service',
 'occupation_ Priv-house-serv',
 'occupation_ Prof-specialty',
 'occupation_ Protective-serv',
 'occupation_ Sales',
 'occupation_ Transport-moving',
 'relationship_ Husband',
 'relationship_ Not-in-family',
 
 ]].values
y=train.target.values
"both dummy and label + scaling"
evaluate_mdl(X,y)
# 0.3141853158547453
# 0.3140829602013742
%%time
scorer=make_scorer(log_loss, greater_is_better=False)

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=0, cv=6,n_jobs=-1 ,scoring=scorer)
grid_search_cv.fit(X, y)
grid_search_cv.best_estimator_

NFOLDS=5
skf = StratifiedKFold(n_splits=NFOLDS)


X_test=test[['age',
 'workclass',
 'fnlwgt',
 'education',
 'education-num',
 'marital-status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'capital-gain',
 'capital-loss',
 'hours-per-week',
 'native-country',
 'workclass_Federal-gov',
 'workclass_Local-gov',
 'workclass_Self-emp-inc',
 'workclass_Self-emp-not-inc',
 'marital-status_ Married-AF-spouse',
 'marital-status_ Married-civ-spouse',
 'marital-status_ Married-spouse-absent',
 'marital-status_ Never-married',
 'marital-status_ Separated',
 'marital-status_ Widowed',
 'occupation_ ?',
 'occupation_ Adm-clerical',
 'occupation_ Craft-repair',
 'occupation_ Exec-managerial',
 'occupation_ Farming-fishing',
 'occupation_ Handlers-cleaners',
 'occupation_ Machine-op-inspct',
 'occupation_ Other-service',
 'occupation_ Priv-house-serv',
 'occupation_ Prof-specialty',
 'occupation_ Protective-serv',
 'occupation_ Sales',
 'occupation_ Transport-moving',
 'relationship_ Husband',
 'relationship_ Not-in-family',
 
 ]].values
preds=np.zeros((X_test.shape[0],))

validations=np.zeros((train.shape[0],))

losses=[]

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    # print(X_train.shape,X_val.shape)
    clf=grid_search_cv.best_estimator_
    clf.fit(X_train,y_train)
    pval=clf.predict_proba(X_val)[:,1]
    loss=log_loss(y_val,pval)
    losses.append(loss)
    validations[val_index]+=pval
    pred_test=clf.predict_proba(X_test)[:,1]
    preds+=pred_test


print(np.mean(losses))
preds=preds/NFOLDS

sub=pd.DataFrame()
sub["uid"]=test.reset_index()["uid"]

sub["target1"]= preds

sub.head()
vals=pd.read_csv("/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv")
vals["predictions1"]=validations


print("val loss : {}".format(log_loss(vals["target"].values,vals["predictions1"].values)))

sub.head()

# sub.to_csv("submit_hyp_opt1.csv",index=None)
sub.rename(columns={"target1":"target"}).to_csv("submission.csv",index=None)
sub.rename(columns={"target1":"target"}).head()
