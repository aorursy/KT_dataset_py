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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')
train.info()

test.info()
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(train['Sex'])

train[['Sex']] = le.transform(train['Sex']) 

train[['Age']] = SimpleImputer().fit_transform(train[['Age']])



p = train.drop(['PassengerId','Name','Ticket','Cabin','Embarked', 'SibSp', 'Parch'], axis=1, inplace =False)



X = p.drop(['Survived'], axis=1)

X = pd.get_dummies(X)

y = p.Survived



train_X, val_x, train_y, val_y = train_test_split(X, y, random_state=42)



print(train_X.shape, train_y.shape)

print(val_x.shape, val_y.shape)


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



def rfc(train_X,train_y,val_X,val_y, n):

    model = RandomForestClassifier(n_estimators=n)

    model.fit(train_X,train_y)

    predict_y = model.predict(val_X)

    return accuracy_score(predict_y, val_y)



n_x = np.arange(100)

a_y = np.arange(100.0)



for n in range(1,100):

    accuracy = rfc(train_X, train_y, val_x, val_y, n)

    n_x[n] = n

    a_y[n] = accuracy

    print("%s estimator's accuracy is %s" %(n, accuracy))

    
import matplotlib.pyplot as plt

plt.plot(n_x, a_y)
'''

from sklearn.model_selection import GridSearchCV

n_estimators = list(range(10,20,1))

min_samples_leaf = list(range(1,10,1))

max_depth = list(range(1,30,1))



#X = pd.concat([train_X,test_X])

#y = pd.concat([train_y, test_y])

grid = GridSearchCV(estimator=RandomForestClassifier(), cv=10, 

                    param_grid=dict(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf, max_depth=max_depth))

grid.fit(X, y)

print(grid.best_score_)

print(grid.best_params_)

'''
'''

from sklearn.ensemble import AdaBoostClassifier



n_estimators = list(range(50,300,10))

learning_rate = np.arange(0.8,1.5,0.1)



grid2 = GridSearchCV(estimator=AdaBoostClassifier(), cv=10,

                    param_grid=dict(n_estimators=n_estimators, learning_rate=learning_rate))

#AdaBoost = AdaBoostClassifier(n_estimators=100,learning_rate=1.3,algorithm='SAMME')

#AdaBoost.fit(train_X,train_y)

#prediction = AdaBoost.score(test_X,test_y)

#print('The accuracy is: ',prediction)



grid2.fit(X, y)

print(grid2.best_score_)

print(grid2.best_params_)

'''
'''

rdfc=RandomForestClassifier(n_estimators=11,min_samples_leaf=2, max_depth=24)

rdfc.fit(X, y)

print(test.shape)

q = test.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace =False)

print(q.shape)

#q = q.dropna(subset=['Fare'], axis=0)

q[['Fare', 'Age']] = SimpleImputer().fit_transform(q[['Fare', 'Age']])

q = pd.get_dummies(q)

idx = q.PassengerId



q = q.drop(['PassengerId'], axis=1)

pred=rdfc.predict(q)

c = pd.concat([pd.DataFrame(idx), pd.DataFrame(pred)], axis=1, join='inner')

c.columns=['PassengerId', 'Survived']

c.to_csv('submission2.csv', index=False, header=True) 

'''
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



'''

md = xgb.XGBClassifier()

md.fit(train_X, train_y)

prediction = md.predict(val_x)

print('The accurasy is:', accuracy_score(prediction, val_y))

'''

n_estimators=list(range(5,15,1))

learning_rate=np.arange(0.1, 1.0, 0.1)

max_depth=list(range(10,30,2))



grid3 = GridSearchCV(estimator=xgb.XGBClassifier(), cv=10,

                    param_grid=dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth))

grid3.fit(X, y)

print(grid3.best_score_)

print(grid3.best_params_)

md = xgb.XGBClassifier(n_estimators=10, learning_rate=0.4, max_depth=10)

md.fit(X, y)

q = test.drop(['Name','Ticket','Cabin','Embarked','SibSp', 'Parch'], axis=1, inplace =False)

q[['Sex']] = le.transform(test['Sex'])

q[['Fare', 'Age']] = SimpleImputer().fit_transform(q[['Fare', 'Age']])

q = pd.get_dummies(q)

idx = q.PassengerId



q = q.drop(['PassengerId'], axis=1)

pred=md.predict(q)

c = pd.concat([pd.DataFrame(idx), pd.DataFrame(pred)], axis=1, join='inner')

c.columns=['PassengerId', 'Survived']

c.to_csv('submission5.csv', index=False, header=True) 
