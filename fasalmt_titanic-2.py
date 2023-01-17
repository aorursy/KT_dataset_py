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
# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.plot.scatter('Fare','Age')
def pre_pro(data,drop_col):
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['FareN'],cat=pd.qcut(data['Fare'],5,retbins=True,duplicates ='drop')
    data.loc[data['Fare']<=cat[0],'Fare']=0
    data.loc[(data['Fare']>cat[0]) & (data['Fare']<=cat[1]) ,'Fare']=1
    data.loc[(data['Fare']>cat[1]) & (data['Fare']<=cat[2]),'Fare']=2
    data.loc[(data['Fare']>cat[2]) & (data['Fare']<=cat[3]),'Fare']=3
    data.loc[(data['Fare']>cat[3]) & (data['Fare']<=cat[4]),'Fare']=4
    data.loc[(data['Fare']>cat[4]) ,'Fare']=5
    data['Fare']=data['Fare'].astype(int)
    data['Family']=data['SibSp']+data['Parch']+1
    data['HasCabin']=1
    data.loc[data['Cabin'].isna(),'HasCabin']=0
    data.loc[data['Embarked'].isna(),'Embarked']=data['Embarked'].mode()
    data['SexP']=data['Sex'].map({'male':1,'female':2})
    age_builder=data[~data['Age'].isna()].groupby(['Fare','Family'])['Age'].mean()
    age_builder = age_builder.to_frame().reset_index()
    for i in age_builder['Fare']:
        for j in age_builder['Family']:
            data.loc[(data['Age'].isna()) & (data['Fare'].values==i) & (data['Family'].values==j),'Age']=age_builder.loc[(age_builder['Fare'].values==i) & (age_builder['Family'].values==j),'Age'].values
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['AgeN'],cat=pd.qcut(data['Age'],5,retbins=True,duplicates ='drop')
    data.loc[data['Age']<=cat[0],'Age']=0
    data.loc[(data['Age']>cat[0]) & (data['Age']<=cat[1]) ,'Age']=1
    data.loc[(data['Age']>cat[1]) & (data['Age']<=cat[2]),'Age']=2
    data.loc[(data['Age']>cat[2]) & (data['Age']<=cat[3]),'Age']=3
    data.loc[(data['Age']>cat[3]) & (data['Age']<=cat[4]),'Age']=4
    data.loc[(data['Age']>cat[4]) ,'Age']=5
    data['Age']=data['Age'].astype(int)
    data['Embarked'].fillna('N', inplace=True)
    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3,'N':4})
    data['Embarked']=data['Embarked'].astype(int)
    data=data.drop(drop_col,axis=1)
    return data,cat
    
drop=['PassengerId','Name','Sex','Ticket','Cabin','FareN','AgeN']
final_train,cat=pre_pro(train,drop)
final_test,cat=pre_pro(test,drop)

colormap = plt.cm.bone
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(final_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


final_train.head()
def model_bulding(model,final_train,target,FS=0):
    if FS==1:
        x=final_train.drop(target,axis=1)
        y=final_train[target]
        sc=StandardScaler()
        x=sc.fit_transform(x)
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.3)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
        average_precision = average_precision_score(yte, ypred)
        precision, recall, _ = precision_recall_curve(yte, ypred)
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    elif FS==0:
        x=final_train.drop(target,axis=1)
        y=final_train[target]
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.3)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
        average_precision = average_precision_score(yte, ypred)
        precision, recall, _ = precision_recall_curve(yte, ypred)
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))        
    
    print("ROC is : ", roc)
    print("Accuracy is : ",acc)
    print("Cross validation score :", accuracies.mean(),accuracies.std())
    
        
def model_bulding_lite(model,final_train,target,FS=0):
    if FS==1:
        x=final_train.drop(target,axis=1)
        y=final_train[target]
        sc=StandardScaler()
        x=sc.fit_transform(x)
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.3)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
        print(classification_report(yte,ypred))
    elif FS==0:
        x=final_train.drop(target,axis=1)
        y=final_train[target]
        xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.3)
        model.fit(xtr,ytr)
        ypred=model.predict(xte)
        ypred_full=model.predict(x)
        roc=roc_auc_score(yte,ypred)
        acc=accuracy_score(yte,ypred)
        accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
        print(classification_report(yte,ypred))
        
    print("ROC is : ", roc)
    print("Accuracy is : ",acc)
    print("Cross validation score :", accuracies.mean(),accuracies.std())
    
        
tar='Survived'
LR=LogisticRegression()
NB=GaussianNB()
RFC=RandomForestClassifier()
svc_plain=SVC()
svc=SVC(C= 10,gamma= 0.1, kernel= 'rbf')
ABC=AdaBoostClassifier(n_estimators=500,learning_rate= 0.75)
xgb=XGBClassifier()
knn=KNeighborsClassifier()
def param_tun(model,param_grid,x,y):
    grid=GridSearchCV(model,param_grid,refit = True, verbose=1,n_jobs=-1)
    grid.fit(x,y)
    return grid.best_params_    
params={'lr':{'C':[1,10,100,1000]},
        'rfc':{'n_estimators':[10,50,100,200,500],'max_features':['auto','log2',0.25,0.5,0.75],'max_depth':[3,5,7,10]},
        'svc':{'C':[10],'gamma': [0.1], 'kernel': ['rbf']},
        'knn':{'n_neighbors':[2,5,7,10,15]},
        'gb':{'learning_rate':[.01,.05,.1,.2,.5],'n_estimators':[50,100,300,500],'max_depth':[3,5,7,10]}
       }

LR=LogisticRegression()
RFC=RandomForestClassifier()
SVC_M=SVC()
KNN=KNeighborsClassifier()
GB=GradientBoostingClassifier()
models={'lr':LR,'rfc':RFC,'svc':SVC_M,'knn':KNN,'gb':GB}
       

tuned_params=pd.DataFrame(columns=['mod','par'])
for k in models.keys():
    df_temp=pd.DataFrame([[k,param_tun(models[k],params[k],final_train.drop('Survived',axis=1),final_train['Survived'])]],columns=['mod','par'])
    print(df_temp)
    tuned_params=tuned_params.append(df_temp)
    print("fitting over for ", k)
    
    
pd.set_option('display.max_colwidth', -1)
tuned_params
LR=LogisticRegression(C=1)
RFC=RandomForestClassifier(max_depth= 5, max_features= 0.75, n_estimators= 100)
SVC_M=SVC(C= 10, gamma= 0.1, kernel= 'rbf')
KNN=KNeighborsClassifier(n_neighbors= 10)
GB=GradientBoostingClassifier(learning_rate= 0.05, max_depth= 3, n_estimators= 300)
models={'lr':LR,'rfc':RFC,'svc':SVC_M,'knn':KNN,'gb':GB}
model_bulding_lite(LR,final_train,tar)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
model_bulding_lite(GB,final_train,tar)
lrr=LR.coef_
rfc=RFC.feature_importances_
gbb=GB.feature_importances_
col=final_train.loc[:,final_train.columns != 'Survived'].columns.values
feature_imp=pd.DataFrame({'col':col,
                         'rfc':rfc,
                         'gb':gbb})
feature_imp
Xtrain=final_train.drop('Survived',axis=1)
Ytrain=final_train['Survived']
def hold_out_pred(model,xtr,ytr,nfold):
    ntrain=xtr.shape[0]
    kf=KFold(ntrain,n_folds=nfold,random_state=0)
    oof_train=np.zeros((xtr.shape[0],))
    
    for i, (train_index,test_index) in enumerate(kf):
        xtr_oo=xtr.iloc[train_index]
        ytr_oo=ytr.iloc[train_index]
        xte_oo=xtr.iloc[test_index]
        model.fit(xtr_oo,ytr_oo)
        oof_train[test_index]=model.predict(xte_oo)
    return oof_train
    
lr_oo_train=hold_out_pred(LR,Xtrain,Ytrain,5)
rfc_oo_train=hold_out_pred(RFC,Xtrain,Ytrain,5)
scv_oo_train=hold_out_pred(SVC_M,Xtrain,Ytrain,5)
knn_oo_train=hold_out_pred(KNN,Xtrain,Ytrain,5)
gb_oo_train=hold_out_pred(GB,Xtrain,Ytrain,5)

L1_pred=np.stack((lr_oo_train,rfc_oo_train,scv_oo_train,knn_oo_train,gb_oo_train))
L1=L1_pred.transpose()
L1_predp=pd.DataFrame(L1,columns=['lr','rfc','svc','knn','gb'])
xgb=XGBClassifier(gamma= 0.5,learning_rate= 0.01,max_depth= 2,min_child_weight= 2,n_estimators= 500)
# xgb_params={'learning_rate':[0.01,.05,.1,.5],'n_estimators':[500,1000,2500,4000,7000],'max_depth':[2,4,7,10],
#             'min_child_weight':[2,4,7],'gamma':[.5,0.9,1.4]}
# xgb_best=param_tun(xgb,xgb_params,L1_predp,Ytrain)
xgb_best
xgb_train_oo=hold_out_pred(xgb,L1_predp,Ytrain,5)
print(classification_report(Ytrain,xgb_train_oo))
print(accuracy_score(y_true=Ytrain,y_pred=xgb_train_oo))

lr_test=LR.predict(final_test)
rfc_test=RFC.predict(final_test)
svc_test=SVC_M.predict(final_test)
knn_test=KNN.predict(final_test)
gb_test=GB.predict(final_test)

oo_test=np.stack((lr_test,rfc_test,svc_test,knn_test,gb_test))
oo_test=oo_test.transpose()
oo_testpd=pd.DataFrame(oo_test,columns=['lr','rfc','svc','knn','gb'])
final_stack2=xgb.predict(oo_testpd)

test_pred=LR.predict(final_test)
pid=test['PassengerId']
final_stack2 = pd.DataFrame({ 'PassengerId': pid,
                            'Survived': final_stack2 })
final_stack2.head()
final_stack2.to_csv('final_stack2.csv', index=False)