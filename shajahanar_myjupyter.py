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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
ttt = pd.read_csv('../input/test.csv')
train.head()
test.tail()
train = train.drop(['Cabin','Ticket','Fare','Name','PassengerId'],axis=1)
train
test = test.drop(['Cabin','Ticket','Fare','Name','PassengerId'],axis=1)
test
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)



plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=test,palette='winter')
def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):

        if Pclass == 1:
            if Sex == 'male':
                return 42.5
            else:
                return 41

        elif Pclass == 2:
            if Sex == 'male':
                return 28
            else:
                return 24

        else:
            if Sex == 'male':
                return 24
            else:
                return 22.5

    else:
        return Age
def impute_age_train(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):

        if Pclass == 1:
            if Sex == 'male':
                return 40
            else:
                return 35

        elif Pclass == 2:
            if Sex == 'male':
                return 30
            else:
                return 28

        else:
            if Sex == 'male':
                return 25
            else:
                return 22

    else:
        return Age
train['Age'] = train[['Age','Pclass','Sex']].apply(impute_age_train,axis=1)
test['Age'] = test[['Age','Pclass','Sex']].apply(impute_age_test,axis=1)

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.head()
test.head()

sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
test.drop(['Sex','Embarked'],axis=1,inplace=True)
test.info()

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train.info()
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
test
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(test)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# acc_log
# from sklearn.svm import SVC
# svc = SVC()
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(test)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# acc_svc
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, y_train)
# Y_pred = knn.predict(test)
# acc_knn = round(knn.score(X_train, y_train) * 100, 2)
# acc_knn
# from sklearn.tree import DecisionTreeClassifier
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# Y_pred = decision_tree.predict(test)
# acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
# acc_decision_tree
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(test)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
survival_prediction = Y_pred
submission = pd.DataFrame({
        "PassengerId": ttt["PassengerId"],
        "Survived": survival_prediction
    })
submission.to_csv('submission.csv', index=False)
submission
