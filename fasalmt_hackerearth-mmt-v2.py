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
import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
%matplotlib inline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split,KFold
from sklearn.model_selection import GridSearchCV,cross_val_score,RandomizedSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')
from math import sqrt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,auc,accuracy_score,precision_recall_curve,mean_squared_error,average_precision_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pdb
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
colormap = plt.cm.bone
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train[['B','C','H','K','N','O','P']].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
def pre_pro(train,test, target, drop_col):
    train['is_train']=1
    test['is_train']=0
    train_y=train[target]
    test_id=test['id']
    test.drop(drop_col, axis=1, inplace=True)
    train.drop([target,drop_col], axis=1, inplace=True)
    com=pd.concat([train, test])
    
    cont=com.select_dtypes(['int64','float64']).columns
    cat=com.select_dtypes(exclude=['int64','float64']).columns
    print("Total null values before treatment : ",com.isna().sum().sum())
    for k in cont:
        com[k].fillna(com[k].mean(), inplace=True)
    for j in cat:
        com[j].fillna(com[j].mode()[0], inplace=True)
    print("Total null values after treatment : ",com.isna().sum().sum())
    com_new=pd.get_dummies(com[cat])
    com_nn=pd.concat([com, com_new], axis=1)
    print(com.columns)
    com_nn.drop(cat, axis=1, inplace=True)
    
    train=com_nn[com_nn['is_train']==1]
    train.drop('is_train', axis=1, inplace=True)
    test=com_nn[com_nn['is_train']==0]
    test.drop('is_train', axis=1, inplace=True)
    print(train.head())
    return train, train_y,test,test_id

target='P'
drop='id'
train_p,train_y,test_p,test_id=pre_pro(train,test, target,drop)
train_p.head()
def model_bulding_lite(model,final_train,target,FS=0):
    if FS==1:
        x=final_train
        y=target
        sc=StandardScaler()
        x=sc.fit_transform(x)
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.3)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y,scoring='roc_auc', cv = 10)
        print(classification_report(yte,ypred))
    elif FS==0:
        x=final_train
        y=target
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.2)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        ypred_full=model.predict(x)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y,scoring='roc_auc', cv = 5)
        print(classification_report(yte,ypred))
        
    print("ROC is : ", roc)
    print("Accuracy is : ",acc)
    print("Cross validation score :", accuracies.mean(),accuracies.std())
   
    
        


# tar='Survived'
# LR=LogisticRegression(C=.12)
# NB=GaussianNB()
# RFC=RandomForestClassifier()
# DC=DecisionTreeClassifier()
# svc_plain=SVC()
# svc=SVC(C= 10,gamma= 0.1, kernel= 'rbf')
# ABC=AdaBoostClassifier(n_estimators=500,learning_rate= 0.75)
# xgb=XGBClassifier()
# knn=KNeighborsClassifier()
# models={'lr':LR,'dc':DC,'rfc':RFC,'svc':svc_plain,'knn':knn,'gb':xgb}
# c=[.07,.08,.09,.1,.11,.12,.14,.16,.175,.2,.25]
# roc=[]
# for i in c:
#     print(i)
#     LR=LogisticRegression(C=i)
#     roc.append(model_bulding_lite(LR,train_p,train_y,FS=0))
    
# sc=pd.DataFrame([c,roc])
# sc=sc.T
# sc.columns=['c', 'roc']
# sc.columns
# sns.jointplot(x='c',y='roc', data=sc)
def param_tuner(model, param_grid, x,y):
    grid=GridSearchCV(model, cv=10, scoring='roc_auc', param_grid=param_grid, n_jobs=-1)
    grid.fit(x,y)
    return grid.best_params_        
model_bulding_lite(LR,train_p,train_y,FS=0)
LR=LogisticRegression(C=.12)
LR.fit(train_p, train_y)
y_train_p=LR.predict_proba(train_p)
probs=pd.concat([pd.DataFrame(train_y),pd.DataFrame(y_train_p)], axis=1)
probs.columns=['p', 'zero', 'one']
sns.lmplot(x='one', y='one', data=probs[(probs['one']<.55) & (probs['one']>.18)], hue='p', fit_reg=False )
xgb=XGBClassifier(learning_rate=0.2, max_depth= 7, n_estimators= 100)
model_bulding_lite(xgb,train_p,train_y)
# param={'learning_rate':[.01,.05,.1,.2,.5],'n_estimators':[50,100,300,500],'max_depth':[3,5,7,10]}
# param_tuner(xgb,param,train_p,train_y)
y_prob_xgb=xgb.predict_proba(train_p)
probs_xgb=pd.concat([pd.DataFrame(train_y), pd.DataFrame(y_prob_xgb)], axis=1)
probs_xgb.columns=['P','one','two']
sns.lmplot(x='one', y='two', hue='P',data=probs_xgb, fit_reg=False)
y_pred=LR.predict(test_p)
test_pred=pd.DataFrame([test_id,y_pred.ravel()])
test_pred=test_pred.T
test_pred.columns=['id','P']
test_pred.head()
test_pred.to_csv('mmt_pred1.csv', index=False)
y_pred_xgb=xgb.predict(test_p)
test_pred_xgb=pd.concat([pd.DataFrame(test_id), pd.DataFrame(y_pred_xgb)], axis=1)
test_pred_xgb.to_csv('mmt_pred2.csv', index=False)







