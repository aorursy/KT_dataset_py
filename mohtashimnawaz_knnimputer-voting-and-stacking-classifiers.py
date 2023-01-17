import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print("Shape of train data:",train.shape)

print("Shape of test data:",test.shape)
total = train.append(test)

print(total.shape)
# Missing values heatmap

sns.heatmap(total.isnull())
# It is clear that there are missing values, let's see which fields have missing values and how much

s = total.isnull().sum()

s[s>0]
# Let's see some columns

total.head()
# Let's first drop some non necessary fields like name and Ticket

train.drop(['Ticket','Name','PassengerId'],inplace=True, axis=1)

test.drop(['Ticket','Name','PassengerId'],inplace=True, axis=1)

print(train.columns)

print(test.columns)
train.head()
# Let's fill Cabin with 1 if there is a cabin or 0 if there is null

train.loc[~train['Cabin'].isnull(),'Cabin']=1

train.loc[train['Cabin'].isnull(),'Cabin']=0

test.loc[~test['Cabin'].isnull(),'Cabin']=1

test.loc[test['Cabin'].isnull(),'Cabin']=0
train.isnull().sum()
test.isnull().sum()
train['Embarked'].value_counts()
train.loc[train['Sex']=='male', 'Sex']=1

train.loc[train['Sex']=='female', 'Sex']=0



test.loc[test['Sex']=='male', 'Sex']=1

test.loc[test['Sex']=='female', 'Sex']=0



train.loc[train['Embarked']=='S', 'Embarked']=0

train.loc[train['Embarked']=='C', 'Embarked']=1

train.loc[train['Embarked']=='Q', 'Embarked']=2



test.loc[test['Embarked']=='S', 'Embarked']=0

test.loc[test['Embarked']=='C', 'Embarked']=1

test.loc[test['Embarked']=='Q', 'Embarked']=2
train.head()
test.head()
y = train['Survived']

train.drop('Survived',axis=1,inplace=True)
# Now all features are numeric, we can apply KNNImputer

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=1)

imputer.fit(train)
train = imputer.transform(train)

test = imputer.transform(test)
total.drop(['Ticket','Name','PassengerId','Survived'],axis=1,inplace=True)

total.head()
train = pd.DataFrame(train, columns = total.columns)

test = pd.DataFrame(test,columns = total.columns)
# Family = SibSp+Parch

train['Family']=train['SibSp']+train['Parch']

test['Family']=test['SibSp']+test['Parch']
train = pd.concat([train,y],axis=1)

train.head()
sns.heatmap(train.corr())
# male=1, female=0

sns.countplot(x=train['Survived'],hue=train['Sex'])
sns.countplot(x=train['Survived'],hue=train['Pclass'])
plt.figure(figsize=(20,6))

ax1=plt.subplot(1,2,1)

ax2=plt.subplot(1,2,2)

sns.violinplot(x=train['Survived'], y=train['Age'],ax=ax1)

sns.violinplot(x=train['Survived'], y=train['Age'],hue=train['Sex'],ax=ax2)

plt.show()
# Let's get dummies for categorical data

# Let's first combine train and test

total = train.append(test)

total.head()
sns.distplot(total['Age'].where(total['Survived']==0))
def convertAge(total):

    total.loc[total['Age']<5,'Age']=1

    total.loc[(total['Age']>=5) & (total['Age']<10),'Age']=2

    total.loc[(total['Age']>=10) & (total['Age']<20),'Age']=3

    total.loc[(total['Age']>=20) & (total['Age']<40),'Age']=4

    total.loc[(total['Age']>=40) & (total['Age']<60),'Age']=5

    total.loc[(total['Age']>=60),'Age']=6

    

    return total

    

total=convertAge(total)
total['Age'].value_counts()
# Let's make Pclass, Sex, Cabin, Embarked to strings type because Pandas get_dummies method recognizes string attributes as categorical

cat = {'Pclass':str, 'Sex':str, 'Cabin':str, 'Embarked':str,'Age':str}

total= total.astype(cat)
total.dtypes
total = pd.get_dummies(total, drop_first=True)

total.head()
train = total.iloc[:891]

test = total.iloc[891:]

y = train['Survived']

train.drop('Survived',inplace=True,axis=1)

test.drop('Survived',inplace=True,axis=1)

print(train.shape)

print(test.shape)

print(y.shape)
def normalizer(sr1, sr2):

    mini = sr1.min()

    maxi = sr1.max()

    sr1 = ((sr1-mini)/(maxi-mini))

    sr2 = ((sr2-mini)/(maxi-mini))

    return sr1, sr2

train['Fare'], test['Fare'] = normalizer(train['Fare'],test['Fare'])
train.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train,y,random_state=0,test_size=0.2,stratify = y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



parameters = {'n_neighbors':[i for i in range(1,10)]}



knn = KNeighborsClassifier()

grd = GridSearchCV(estimator=knn, param_grid=parameters, n_jobs=-1, cv=10, return_train_score=True)

grd.fit(X_train,y_train)
grd.best_params_
# This is actually rediculous, model was highly overfitting with neighbors=5

knn = KNeighborsClassifier(n_neighbors=26)

knn.fit(X_train,y_train)

print("Train Score:",knn.score(X_train,y_train))

print("Test Score:",knn.score(X_test,y_test))
from sklearn.linear_model import RidgeClassifier

parameters = {'alpha':[0.01,0.1,1.0,5.0,10,100,1000],'max_iter':[500,1000,1500,2000]}



ridge = RidgeClassifier()

grd = GridSearchCV(estimator=ridge, param_grid=parameters, n_jobs=-1, cv=10, return_train_score=True)

grd.fit(X_train,y_train)
grd.best_params_
# Here model performs significantly okay with CV parameters

ridge = RidgeClassifier(alpha=0.01, max_iter=500)

ridge.fit(X_train,y_train)

print("Train Score:", ridge.score(X_train,y_train))

print("Test Score:", ridge.score(X_test,y_test))
import xgboost as xgb

# xgtrain = xgb.DMatrix(data=X_train, label=y_train)

# xg_clf = xgb.XGBClassifier(max_depth=3)



# parameters = {'learning_rate':[0.01,0.1,1],'alpha':[0.001,0.01,0.1,1.0,10],'lambda':[0.001,0.01,0.1,1.0,10],'n_estimators':[500,1000,1500]}



# grd = GridSearchCV(estimator=xg_clf, param_grid=parameters, n_jobs=-1)

# grd.fit(X_train,y_train)
grd.best_params_
xg_clf = xgb.XGBClassifier(max_depth=3, alpha=5,reg_lambda=10,learning_rate=0.2,n_estimator=1000)

xg_clf.fit(X_train,y_train)

print('Train score:',xg_clf.score(X_train,y_train))

print('Test score:',xg_clf.score(X_test,y_test))
y_ridge_t = ridge.predict(X_test)

y_xg_t = xg_clf.predict(X_test)
from sklearn.metrics import confusion_matrix,f1_score

print("F1-Score Ridge:", f1_score(y_test,y_ridge_t))

print("Ridge Confusion Matrix:")

confusion_matrix(y_test,y_ridge_t)
print("F1-Score Ridge:", f1_score(y_test,y_xg_t))

print("Ridge Confusion Matrix:")

confusion_matrix(y_test,y_xg_t)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

print("NB Train score:",nb.score(X_train,y_train))

print("NB Test score:",nb.score(X_test,y_test))

y_nb_t = nb.predict(X_test)

print("F1-Score Ridge:", f1_score(y_test,y_nb_t))

print("Confusion Matrix:")

confusion_matrix(y_test, y_nb_t)
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('ridge',ridge),('xgb',xg_clf),('nb',nb)],voting='hard')

vc.fit(X_train,y_train)

print("Voting Train score:",vc.score(X_train,y_train))

print("Voting Test score:",vc.score(X_test,y_test))

y_vc_t = vc.predict(X_test)

print("F1-Score Ridge:", f1_score(y_test,y_vc_t))

print("Confusion Matrix:")

confusion_matrix(y_test, y_vc_t)
from sklearn.ensemble import StackingClassifier

stk = StackingClassifier(estimators=[('ridge',ridge),('xgb',xg_clf)],final_estimator=xg_clf)

stk.fit(X_train,y_train)

print("Stacking Train score:",stk.score(X_train,y_train))

print("Stacking Test score:",stk.score(X_test,y_test))

y_stk_t = stk.predict(X_test)

print("F1-Score Ridge:", f1_score(y_test,y_stk_t))

print("Confusion Matrix:")

confusion_matrix(y_test, y_stk_t)
y_final_t = (0.25*y_ridge_t+0.3*y_xg_t+0.2*y_vc_t+0.3*y_stk_t)

mask1 = y_final_t<0.8

mask2 = y_final_t>=0.8

y_final_t[mask1]=0

y_final_t[mask2]=1

y_final_t
confusion_matrix(y_test,y_final_t)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_final_t)
# Let's prepare final models

ridge.fit(train,y)

xg_clf.fit(train,y)

vc.fit(train,y)

stk.fit(train,y)
pred_ridge = ridge.predict(test)

pred_xg = xg_clf.predict(test)

pred_vc = vc.predict(test)

pred_stk = stk.predict(test)

pred_final = (0.2*pred_ridge+0.3*pred_xg+0.25*pred_vc+0.25*pred_stk)

mask1 = pred_final<0.8

mask2 = pred_final>=0.8

pred_final[mask1]=0

pred_final[mask2]=1

pred_final = pred_final.astype(int)
test = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_final})
submission.head()
submission.to_csv('submission.csv',index=False)