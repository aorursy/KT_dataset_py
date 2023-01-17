# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are
# available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train.info()
train.head()
sns.heatmap(train.isnull(),annot=False,yticklabels=False,cbar=False,cmap='viridis')
train.drop(labels='Cabin',inplace=True,axis=1)
test.drop(labels='Cabin',inplace=True,axis=1)


def check_class(x):
    if pd.isnull(x['Age']):
        return pmean[x['Pclass']]
    return x['Age']
pmean=train.groupby('Pclass').mean()['Age']

train['Age']=train.apply(check_class,axis=1)
test['Age']=test.apply(check_class,axis=1)

test['Fare']=test['Fare'].fillna(np.mean(test['Fare'])).astype(float)



sns.heatmap(train.isnull(),cmap='viridis',cbar=False,yticklabels=False,annot=False)
plt.style.use('ggplot')
sns.distplot(train['Age'],bins=20,color='blue')
sns.countplot(data=train,x='Pclass',hue='Survived')
sns.barplot(data=train,x='Survived',y='Age',hue='Sex',palette='coolwarm')
sns.stripplot(data=train,x='Pclass',y='Fare',hue='Survived',jitter=True)
sns.boxplot(data=train,x='Pclass',y='Age',hue='Survived')
sns.violinplot(data=train,y='Fare',x='Sex',hue='Survived',split=True)
sns.jointplot(data=train,y='Age',x='Fare')
sns.lmplot(data=train,x='Age',y='Survived',hue='Pclass',fit_reg=False)

plt.figure(figsize=(10,5))
sns.heatmap(train.corr(),annot=True,cmap='coolwarm')
sns.pairplot(train,hue='Survived')
vtrain=train
vtest=test
train=train.loc[:,['Pclass','Survived','Sex','Age','SibSp','Parch','Fare','Embarked']]
test=test.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
pc=pd.get_dummies(train['Pclass'],drop_first=True,prefix='pclass')
pctest=pd.get_dummies(test['Pclass'],drop_first=True,prefix='pclass')

sex=pd.get_dummies(train['Sex'],drop_first=True,prefix='sex')#male=1 and  female=0
sextest=pd.get_dummies(test['Sex'],drop_first=True,prefix='sex')#male=1 and  female=0

em=pd.get_dummies(train['Embarked'],drop_first=True)#C=1 when Q=0 and S=0\
emtest=pd.get_dummies(test['Embarked'],drop_first=True)#C=1 when Q=0 and S=0
def isSignificant(x):
    for s in x.split(','):
        if '.' in s:
            return s.split(' ')[1]
s=vtrain["Name"].apply(isSignificant)
unique_surnames_train=s.unique()

t=vtest["Name"].apply(isSignificant)
unique_surnames_test=t.unique()

def fill(x):
    for i in range(len(unique_surnames_train)):
        if unique_surnames_train[i] in x:
            return i
extratrain=pd.get_dummies(vtrain["Name"].apply(fill).replace(np.arange(len(unique_surnames_train)),unique_surnames_train),drop_first=True)#capt
extratest=pd.get_dummies(vtest["Name"].apply(fill).replace(np.arange(len(unique_surnames_train)),unique_surnames_train),drop_first=True)#capt

temp=pd.concat([extratest,pd.DataFrame(np.zeros((test.shape[0],9)),columns=['Col.','Don.','Jonkheer.','Lady.','Major.','Mlle.','Mme.','Sir.', 'the']).astype('int')],axis=1)

extratest=temp.loc[:,extratrain.columns]
def hasalpha(x):
    for i in x:
        if str.isalpha(i):
            return i
    return 'non'
extr=pd.get_dummies(vtrain['Ticket'].apply(hasalpha)).iloc[:,:-1]
exte=pd.get_dummies(vtest['Ticket'].apply(hasalpha)).iloc[:,:-1].loc[:,extr.columns]
train=pd.concat([train,pc,sex,em],axis=1).drop(['Pclass','Sex','Embarked'],axis=1)

test=pd.concat([test,pctest,sextest,emtest],axis=1).drop(['Pclass','Sex','Embarked'],axis=1)
dftr=pd.read_csv('../input/train.csv')
dfte=pd.read_csv('../input/test.csv')

def app(x):
    if pd.notnull(x):
        return x[0]
    return 
xx=pd.get_dummies(dftr['Cabin'].apply(app)).iloc[:,:-2]

xte=pd.get_dummies(dfte['Cabin'].apply(app))
xte['T']=np.zeros((len(xte),1)).astype('int')
xte=xte.iloc[:,:-2]
train=pd.concat([train,extratrain,extr,xx],axis=1)
test=pd.concat([test,extratest,exte,xte],axis=1)

xtrain=train.iloc[:,1:].values
xtest=test.values
ytrain=train.iloc[:,0].values

from sklearn.preprocessing import  StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.transform(xtest)

from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression(C=1,solver='saga')
regressor.fit(xtrain,ytrain)
from sklearn.svm import SVC
regressor2=SVC(C=1,gamma=0.01,kernel='rbf')

regressor2.fit(xtrain,ytrain)
from sklearn.ensemble import RandomForestClassifier
regressor3=RandomForestClassifier(criterion='entropy',n_estimators=500)
regressor3.fit(xtrain,ytrain)
regressor3.score(xtrain,ytrain)
test['Survived']=regressor.predict(xtest)
from sklearn.model_selection import  GridSearchCV
param_svm=[
    {
        'kernel':['rbf'],
         'C':[0.1,0.01,1,5,10],
        'gamma':[1,0.1,0.01]
    },
     {
        'kernel':['linear'],
          'C':[0.1,0.01,1,5,10],
        'gamma':[1,0.1,0.01,5,10]

    },
    {
         'kernel':['sigmoid'],
          'C':[0.1,0.01,1,5,10],
        'gamma':[1,0.1,0.01]
    }

]

param_rf=[
    
    { 'n_estimators':[10,100,300,600,500], 'criterion':['gini'],'max_depth':[2,5,10,20,50,70,100,150],'max_leaf_nodes':[2,5,10,20,50]},
    { 'n_estimators':[10,100,300,600,500], 'criterion':['entropy'],'max_depth':[2,5,10,20,50,70,100,150],'max_leaf_nodes':[2,5,10,20,50]}

]
param_logistic=[
    {'solver':['newton-cg'],
      'C':[10,0.1,0.01,1]
     }, 
    {'solver':['lbfgs'],
     'C':[10,0.1,0.01,1]
    },
    {'solver':['liblinear'],
     'C':[10,0.1,0.01]
    },
    {'solver':['sag'],'C':[10,0.1,0.01,1]
    }    
]

gc_svm=GridSearchCV(regressor2,param_grid=param_svm,scoring='accuracy',cv=10).fit(xtrain,ytrain)
gc_logistic=GridSearchCV(regressor,param_grid=param_logistic,scoring='accuracy',cv=10).fit(xtrain,ytrain)
gc_rf=GridSearchCV(regressor3,param_grid=param_rf,scoring='accuracy',cv=10).fit(xtrain,ytrain)



regressor3=gc_rf.best_estimator_
regressor3.fit(xtrain,ytrain)


print("SVM")
print(gc_svm.best_params_)
print('-----------------------------------------------------------------------------------')
print('logistic regression')
print(gc_logistic.best_params_)
print('-----------------------------------------------------------------------------------')
print('random forest')
print(gc_rf.best_params_)
print("SVM")
print(gc_svm.best_score_)
print('-----------------------------------------------------------------------------------')
print('logistic regression')
print(gc_logistic.best_score_)
print('-----------------------------------------------------------------------------------')
print('random forest')
print(gc_rf.best_score_)
output_train=pd.DataFrame([regressor.predict(xtrain),regressor2.predict(xtrain)]).apply(lambda x: x.mode()).iloc[0].values
output=pd.DataFrame([regressor.predict(xtest),regressor2.predict(xtest)]).apply(lambda x: x.mode())
from sklearn.metrics import classification_report
print(classification_report(ytrain,output_train))

submission=pd.read_csv('../input/gender_submission.csv')
submission['Survived']=output.iloc[0].values

# output.iloc[0]
submission.to_csv('finaloutput.csv',index=False)

