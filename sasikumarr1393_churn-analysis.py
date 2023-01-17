# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
churn_data = pd.read_csv("../input/watson.csv")

churn_data.head()
churn_data.shape
churn_data.info()
churn_data.describe()
#dropping customerID 

churn_data = churn_data.drop('customerID',axis=1)

churn_data.isna().sum()
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'],errors='coerce')

churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].median(),inplace=True)
churn_data.info()
churn_objdata = churn_data.select_dtypes(include='object')

churn_numdata = churn_data.select_dtypes(include=['int','float'])
churn_objdata.head()
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler

churn_objdata=churn_objdata.apply(LabelEncoder().fit_transform)
churn_objdata.shape
churn_objdata['InternetService'].value_counts()
churn_numdata.head()
lis_num = ['tenure','MonthlyCharges','TotalCharges']

minmax=MinMaxScaler()

for i in lis_num:

    churn_numdata[i+'_norm'] = minmax.fit_transform(np.array(churn_numdata[i]).reshape(-1,1))

    
churn_numdata.drop(['tenure','MonthlyCharges','TotalCharges'],axis=1,inplace=True)

churn_numdata.head()
final_data = pd.concat([churn_numdata,churn_objdata],axis=1)
final_data.head()
final_data.corr()['Churn']
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,auc,accuracy_score,precision_score,recall_score
X =final_data.drop('Churn',axis=1)

Y=final_data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=0)
knn=KNeighborsClassifier()

nb=GaussianNB()

dt=DecisionTreeClassifier()

rf=RandomForestClassifier()
for model,name in zip([knn,nb,dt,rf],['KNN','NB','DT','RF']):

    model.fit(xtrain,ytrain)

    ypred=model.predict(xtest)

    fpr,tpr,_=roc_curve(ypred,ytest)

    print("AUC score: %0.03f[%s]"%(auc(fpr,tpr),name))

    print("Accuracy : %0.03f[%s]"%(accuracy_score(ypred,ytest),name))

    print("Precesion: %0.03f[%s]"%(precision_score(ypred,ytest),name))

    print("Recall: %0.03f[%s]"%(recall_score(ypred,ytest),name))
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}

GS=GridSearchCV(knn,params,cv=5,scoring='f1_weighted')

GS.fit(X,Y)
GS.best_params_
Knn_reg=KNeighborsClassifier(n_neighbors=41,weights='uniform')

Knn_reg.fit(xtrain,ytrain)

Ypred= Knn_reg.predict(xtest)

accuracy_score(Ypred,ytest)
from sklearn.model_selection import GridSearchCV

rf_params = {'n_estimators':np.arange(1,25),'criterion':['gini','entropy']}

GS=GridSearchCV(rf,rf_params,cv=5)

GS.fit(X,Y)
GS.best_params_
rf_reg = RandomForestClassifier(criterion='gini',n_estimators=82)

rf_reg.fit(xtrain,ytrain)

y_pred = rf_reg.predict(xtest)

accuracy_score(y_pred,ytest)
from sklearn.model_selection import GridSearchCV

dt_params = {'max_depth':np.arange(1,100),'criterion':['gini','entropy']}

GS=GridSearchCV(dt,dt_params,cv=5)

GS.fit(X,Y)
GS.best_params_
dt_reg = DecisionTreeClassifier(criterion='gini',max_depth=5)

dt_reg.fit(xtrain,ytrain)

y_pred = dt_reg.predict(xtest)

accuracy_score(y_pred,ytest)
from sklearn.ensemble import BaggingClassifier
knn_bagged = BaggingClassifier(Knn_reg,n_estimators=50)

dt_bagged = BaggingClassifier(dt_reg,n_estimators=50)

nb_bagged= BaggingClassifier(nb,n_estimators=50)
from sklearn.model_selection import KFold

from sklearn import metrics

kf=KFold(n_splits=5,shuffle=True,random_state=0)

for model,name in zip([Knn_reg,nb,dt_reg,rf_reg,knn_bagged,dt_bagged,nb_bagged],['KNN_Regularized','NB','DT_Regularized','RF_Regularized',

                                                                                 'KNN_bagged','DT_bagged','NB_bagged']):

    auc_score=[]

    for train_idx,test_idx in kf.split(X,Y):

        xtrain,xtest=X.iloc[train_idx,:],X.iloc[test_idx,:]

        ytrain,ytest=Y[train_idx],Y[test_idx]

        model.fit(xtrain,ytrain)

        ypred=model.predict(xtest)

        accuracy=accuracy_score(ypred,ytest)

        auc_score.append(accuracy)

    print("AUC score: %0.03f(+/- %0.06f)[%s]"%(np.mean(auc_score),np.var(auc_score,ddof=1),name))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

nb_boost =AdaBoostClassifier(nb,n_estimators=50)

dt_reg_boost = AdaBoostClassifier(dt_reg,n_estimators=50)

rf_reg_boost = AdaBoostClassifier(rf_reg,n_estimators=50)

dt_bagged_boost =AdaBoostClassifier(dt_bagged,n_estimators=50)

nb_bagged_boost = AdaBoostClassifier(nb_bagged,n_estimators=50)

grad_boost = GradientBoostingClassifier(n_estimators=50)
from sklearn.model_selection import KFold

kf=KFold(n_splits=5,shuffle=True,random_state=0)

for model,name in zip([nb_boost,dt_reg,dt_reg_boost,rf_reg,rf_reg_boost,dt_bagged,nb_bagged,dt_bagged_boost,nb_bagged_boost,grad_boost],

                      ['NB_boost','DT_Regularized','DT_Reguralized_boosted','RF_Regularized','RF_Regularized_boosted','DT_bagged',

                       'NB_bagged','DT_bagged&boosted','NB_bagged&boosted','GradientBoosting']):

    auc_score=[]

    for train_idx1,test_idx1 in kf.split(X,Y):

        xtrain1,xtest1=X.iloc[train_idx1,:],X.iloc[test_idx1,:]

        ytrain1,ytest1=Y[train_idx1],Y[test_idx1]

        model.fit(xtrain1,ytrain1)

        ypred=model.predict(xtest1)

        accuracy=accuracy_score(ypred,ytest1)

        auc_score.append(accuracy)

    print("AUC score: %0.03f(+/- %0.06f)[%s]"%(np.mean(auc_score),np.var(auc_score,ddof=1),name))
from sklearn.ensemble import VotingClassifier

stacked=VotingClassifier(estimators=[('GradientBoosting',grad_boost),('DT_bagged',dt_bagged)])
acc_score=[]

for train_idx2,test_idx2 in kf.split(X,Y):

    xtrain,xtest=X.iloc[train_idx2,:],X.iloc[test_idx2,:]

    ytrain,ytest=Y[train_idx2],Y[test_idx2]

    stacked.fit(xtrain,ytrain)

    ypred=stacked.predict(xtest)

    accuracy=accuracy_score(ypred,ytest)

    acc_score.append(accuracy)

print('Accuracy score : %0.3f(+/- %0.06f)'% (np.mean(acc_score),np.var(acc_score)))