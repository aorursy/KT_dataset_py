#importing nessesary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mssno
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
%matplotlib inline
#loading data
train = pd.read_csv('../input/train.csv', na_values=-1,)
test = pd.read_csv('../input/test.csv', na_values=-1)
print('the shape of train set is', train.shape)
print('the shape of train set is', test.shape)
#dropping columns which has no correlation with target
train_drop = train.columns[train.columns.str.startswith("ps_calc")]
test_drop = test.columns[test.columns.str.startswith("ps_calc")]
train.drop(train_drop, inplace=True, axis=1)
test.drop(test_drop, inplace=True, axis=1)
print('the shape of train set is', train.shape)
print('the shape of train set is', test.shape)
#creating a function with filling nan with a value
def filling_nan(df):
    cols = df.columns
    for col in cols:
        if df[col].isnull().sum()>0:
            df[col].fillna(df[col].mode()[0],inplace=True)
            
filling_nan(train)
filling_nan(test)
#making some lists to 
category_cols = [col for col in train.columns if '_cat' in col]
binary_cols = [col for col in train.columns if '_bin' in col]
other_col = [col for col in train.columns if col not in binary_cols + category_cols]
#defining outliers
num_col = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14']

def outlier(df,columns):
    for col in columns:
        df[np.abs(df[col]-df[col].mean())<=(3*df[col].std())]
        
outlier(train, num_col)
def OHE(df1,df2,column):
    cat_col = column
    len_df1 = df1.shape[0]
    
    df = pd.concat([df1,df2],ignore_index=True)
    c2,c3 = [],{}
    
    print('Categorical feature',len(column))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2
train1, test1 = OHE(train, test, category_cols)
X_train = train1.drop(['target','id'],axis=1)
y_train = train1['target']
X_test = test1.drop(['target','id'],axis=1)

del train1,test1
for column in ["ps_car_02_cat", "ps_car_03_cat", "ps_car_05_cat", "ps_car_07_cat", "ps_car_08_cat", "ps_ind_04_cat"]:
    X_train.drop(column, axis=1, inplace=True)
for column in ["ps_car_02_cat", "ps_car_03_cat", "ps_car_05_cat", "ps_car_07_cat", "ps_car_08_cat", "ps_ind_04_cat"]:
    X_test.drop(column, axis=1, inplace=True)  
kf = StratifiedKFold(random_state=42,shuffle=True)

for train_index,holdout_index in kf.split(X_train,y_train):   
    xtr,xvl = X_train.loc[train_index],X_train.loc[holdout_index]
    ytr,yvl = y_train[train_index],y_train[holdout_index]
gbm = xgb.XGBClassifier(
    learning_rate = 0.01, 
    max_depth = 4,
    colsample_bytree = 0.8,
    subsample = 0.7,
    min_child_weight = 5,
    gamma=0.1, 
    reg_alpha=0.1,
    n_estimators=1000,
    objective= 'binary:logistic', 
    nthread=4,
    scale_pos_weight=1, 
    seed=42,
    n_jobs=4
)

gbm.fit(xtr, ytr)
proba = gbm,predict_proba(xvl)
fpr,tpr, threshold = roc_curve(yvl,proba)
auc_val = auc(fpr,tpr)

plt.figure(figsize=(14,8))
plt.title('Reciever Operating Charactaristics')
plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
pred_test_full=0

gbm.fit(X_train, y_train)
pred_test_full += gbm.predict_proba(X_test)[:,1]

submit = pd.DataFrame({'id':test['id'],'target':pred_test_full})
submit.to_csv('porto_xgboost_without_.csv',index=False) 