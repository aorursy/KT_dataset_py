# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.metrics import cross_validation 

#from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv', header=0)

df_test = pd.read_csv('/kaggle/input/titanic/test.csv', header=0)

df.info()

df_test.info()
cols = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']

df=df.drop(cols, axis=1)

df_test = df_test.drop(cols,axis=1)

df.info()

df_test.info()
df['Age'] = df['Age'].interpolate()

df.info()

df_test['Age'] = df_test['Age'].interpolate()

df_test['Fare'] = df_test['Age'].interpolate()

df_test.info()
dummies=[]

cols=['Pclass', 'Sex', 'Embarked']



for col in cols:

    dummies.append(pd.get_dummies(df[col]))



titanic_dummies = pd.concat(dummies, axis=1)

df = pd.concat((df,titanic_dummies), axis=1)

df.info()



dummies_test=[]

for col in cols:

    dummies_test.append(pd.get_dummies(df_test[col]))



titanic_dummies_test = pd.concat(dummies_test, axis=1)

df_test = pd.concat((df_test,titanic_dummies_test), axis=1)

df_test.info()
df = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)

df_test = df_test.drop(['Sex', 'Embarked', 'Pclass'], axis=1)

df.info()

df_test.info()
X = df.values

y = df['Survived'].values

print(X.shape)

print(y.shape)

X = np.delete(X, [1], axis=1)

print("After deleting array shape =",X.shape)



X_final = df_test.values

print(X_final.shape)

# no survived column in test.csv
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

dtc = tree.DecisionTreeClassifier(max_depth=5)

dtc.fit(X_train,y_train)

dtc.score(X_test,y_test)

# public score of 0.72248
lr = LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_test, y_test)

# public score of 0.75598
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit (X_train, y_train)

rfc.score (X_test, y_test)

#public score of 0.74162
gbc = GradientBoostingClassifier(n_estimators=50)

gbc.fit (X_train, y_train)

gbc.score (X_test, y_test)

#public score of 0.7272
#Bagging



bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)

bg.fit(X_train,y_train)

bg.score(X_test, y_test)



# public score of 0.70813
#Boosting - Ada Boost



adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)

adb.fit(X_train,y_train)

adb.score(X_test, y_test)

# public score of 0.65550
# Voting Classifier - Multiple Model Ensemble 

svm = SVC(kernel = 'poly', degree = 2 )

evc = VotingClassifier( estimators= [('lr',lr),('dtc',dtc),('svm',svm)], voting = 'hard')

evc.fit(X_train,y_train)

evc.score(X_test, y_test)

# public score of 0.76076
y_results = dtc.predict(X_final)

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_results

    })

submission.to_csv('my_titanic_submission.csv', index=False)