!ls ../input/diabetes
%%time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data= pd.read_csv('../input/diabetes/diabetes_readmit_dataset_cleaned.csv')
data.columns = data.columns.str.replace(' ', '')
data["Target"] = data['readmitted_dttm'].map({'NO': 0,'<30': 1})
y = data['Target']
data.drop('Target', axis=1,inplace=True)
data.drop('readmitted_dttm', axis=1,inplace=True)
data.columns
id_var=[]
for i in data.columns:
    if i[-2:].lower()=='id':
        id_var.append(i)

id_var
data['admission_source_id'].value_counts()
var = list(set(data.columns) - set(id_var))
var
cat_var = []
num_var = []
for i in var:
        if data[i].dtype == 'O':
            cat_var.append(i)
        else:
            num_var.append(i)
cat_var
num_var
data['age'].value_counts()
for i in id_var:
    print(i,'--------------',data[i].nunique()/len(data),'---------',data[i].nunique())
data = pd.get_dummies(data,columns= id_var,drop_first=True)
cat_var = list(set(cat_var)-set(id_var))
cat_var
data['glyburide-metformin'].value_counts()
data['pioglitazone'].value_counts()
data['metformin'].value_counts()
for i in var:
    print(i,'----','\n',data[i].value_counts())
for i in cat_var:
    print(i,'--------------',data[i].nunique()/len(data),'---------',data[i].nunique())
hc=[]
lc=[]
for i in cat_var:
    if data[i].nunique() > 10:
        hc.append(i)
    else:
            lc.append(i)
hc
lc
data = pd.get_dummies(data,columns= lc,drop_first=True)
for i in hc:
    data[i]=data[i].map(data[i].value_counts().to_dict())
data.shape
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,y,test_size=.3, random_state = 2020)
x_train.shape, x_test.shape, y_train.shape, y_test.shape 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,auc,roc_curve, roc_auc_score
%%time
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
proba_train_lr = lr.predict_proba(x_train)[:,1]
proba_test_lr = lr.predict_proba(x_test)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_lr))
print('train_auc :', roc_auc_score(y_train,proba_train_lr))
%%time
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
proba_train_dt = dt.predict_proba(x_train)[:,1]
proba_test_dt = dt.predict_proba(x_test)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_dt))
print('test_auc :', roc_auc_score(y_train,proba_train_dt))
%%time
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
proba_train_dt = dt.predict_proba(x_train)[:,1]
proba_test_dt = dt.predict_proba(x_test)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_dt))
print('train_auc :', roc_auc_score(y_train,proba_train_dt))
%%time
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train,y_train)
proba_train_xg = xg.predict_proba(x_train)[:,1]
proba_test_xg = xg.predict_proba(x_test)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_xg))
print('train_auc :', roc_auc_score(y_train,proba_train_xg))
%%time
from catboost import CatBoostClassifier
ct = CatBoostClassifier(silent=True)
ct.fit(x_train,y_train)
proba_train_ct = ct.predict_proba(x_train)[:,1]
proba_test_ct =  ct.predict_proba(x_test)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_ct))
print('train_auc :', roc_auc_score(y_train,proba_train_ct))
%%time
from lightgbm import LGBMClassifier
lt = LGBMClassifier()
lt.fit(x_train.values,y_train)
proba_train_lt = lt.predict_proba(x_train.values)[:,1]
proba_test_lt =  lt.predict_proba(x_test.values)[:,1]
print('test_auc :', roc_auc_score(y_test,proba_test_lt))
print('train_auc :', roc_auc_score(y_train,proba_train_lt))
