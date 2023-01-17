# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
total = train.append(test)
print(total.shape)

total.head()
total.drop(columns=['Cabin'], inplace = True)
total.head()
total.isnull().sum()
total['Embarked'].value_counts()
total.loc[total['Embarked'].isnull()]
temp = total.loc[(total['Pclass']==1) & (total['Survived']==1.0) & (total['Sex']=='female') & (total['SibSp']==0)]
temp['Embarked'].value_counts()
total.loc[total['Embarked'].isnull(),'Embarked']='S'
total.isnull().sum()
total.loc[total['Fare'].isnull()]
total.loc[total['Pclass']==3]['Fare'].mean()
total.loc[total['Fare'].isnull(),'Fare']=13.3
total.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x='Survived', data=train)
sns.countplot(x=total['Survived'], hue = total['Sex'],data=train)
total.head()
total[(total['SibSp']+total['Parch']>0) & (total['Survived']==0.0)].count()
total[(total['SibSp']+total['Parch']>0) & (total['Survived']==1.0)].count()
total[(total['SibSp']+total['Parch']==0) & (total['Survived']==0.0)].count()
total[(total['SibSp']+total['Parch']==0) & (total['Survived']==1.0)].count()
sns.violinplot(x='Survived', y = 'Age', data=train)
sns.violinplot(x='Sex', y = 'Age', data=train)
sns.violinplot(x='Pclass', y = 'Age', data=train)
class1 = total[total['Pclass']==1]['Age'].mean()

class2 = total[total['Pclass']==2]['Age'].mean()

class3 = total[total['Pclass']==3]['Age'].mean()
print(class1,class2,class3)
total.loc[(total['Age'].isnull()) & (total['Pclass']==1), 'Age']=class1

total.loc[(total['Age'].isnull()) & (total['Pclass']==2), 'Age']=class2

total.loc[(total['Age'].isnull()) & (total['Pclass']==3), 'Age']=class3
total.isnull().sum()
total['Relatives'] = total['SibSp']+total['Parch']
total.drop('SibSp', inplace=True, axis=1)

total.drop('Parch', inplace=True,axis=1)
print(train.shape, test.shape)
total.drop('PassengerId', inplace=True, axis=1)

total.drop('Name', inplace=True,axis=1)

total.drop('Ticket', inplace=True, axis=1)
total.head()
#Pclass is actually categorical

total['Pclass'].value_counts()
train = total.iloc[:891,:]

test = total.iloc[891:,:]

print(train.shape, test.shape)
test.drop(['Survived'], inplace=True, axis=1)
test.head()
y = train['Survived']
print(type(y))

y.head()
X = train.drop('Survived', axis=1)
print(type(X))

X.head()
sex = pd.get_dummies(X['Sex'],drop_first=True)

embark = pd.get_dummies(X['Embarked'],drop_first=True)

pclass = pd.get_dummies(X['Pclass'],drop_first=True)
X= pd.concat([X,sex,embark,pclass], axis=1)

X.head()
X.drop(['Sex', 'Embarked', 'Pclass'], inplace=True, axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size=0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state = 42,max_depth=10, min_samples_leaf=4)
clf.fit(X_train, y_train)
clf.score(X_train,y_train)
clf.score(X_test,y_test)
test.head()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

pclass = pd.get_dummies(test['Pclass'],drop_first=True)



test= pd.concat([test,sex,embark,pclass], axis=1)

test.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)

test.head()
prediction = clf.predict(test).astype(int)
test_new = pd.read_csv('/kaggle/input/titanic/test.csv')

test_new.head()
submission = pd.DataFrame({'PassengerId':test_new['PassengerId'],'Survived':prediction})
print(submission.shape)

submission.head()
submission.to_csv('submission.csv',index=False)
import xgboost as xgb
dtrain = xgb.DMatrix(X_train,label=y_train)

dtest = xgb.DMatrix(X_test,label=y_test)

param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}

param['nthread'] = 4

param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 12

bst = xgb.train(param, dtrain, num_round, evallist)
xgbtest = xgb.DMatrix(test)
xgb_pred = bst.predict(xgbtest)
type(xgb_pred)
submission_xgb = pd.DataFrame({'PassengerId':test_new['PassengerId'],'Survived':prediction})
print(submission_xgb.shape)

submission_xgb.head()
submission['Survived'].value_counts()
submission_xgb['Survived'].value_counts()
submission_xgb.to_csv('submission_xgb.csv',index=False)
(submission_xgb['Survived']==submission['Survived']).sum()
from catboost import CatBoostClassifier

cat_boost =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=2, learning_rate=0.5, depth=3)
cat_boost.fit(X_train,y_train,eval_set=(X_test,y_test), early_stopping_rounds=100)
cat_boost.score(X_test,y_test)
prediction_cb = cat_boost.predict(test)
submission_cb=pd.DataFrame({'PassengerId':test_new['PassengerId'],'Survived':prediction_cb})
submission_cb.head()
submission_xgb.to_csv('submission_cb.csv',index=False)