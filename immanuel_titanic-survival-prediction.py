# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#imports for algorithm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Any results you write to the current directory are saved as output.
titanic_train_df = pd.read_csv('../input/train.csv')

titanic_test_df = pd.read_csv('../input/test.csv')
#Drop unwanted columns>>>>>>>>>>>



#Training data

train_df = titanic_train_df.drop(['PassengerId','Survived','Name','Cabin','Embarked'],axis=1)



Y_train = titanic_train_df['Survived']



#Testing data

test_df = titanic_test_df.drop(['PassengerId','Name','Cabin','Embarked'],axis=1)
train_df.info()

test_df.info()
# age >> replace NA with median of ages



train_df['Age'] = train_df['Age'].fillna(train_df["Age"].median()).astype(int)

test_df['Age'] = test_df['Age'].fillna(test_df["Age"].median()).astype(int)
# Fare >> replace NA with median of fares



train_df['Fare'] = train_df['Age'].fillna(train_df["Fare"].median()).astype(int)

test_df['Fare'] = test_df['Fare'].fillna(test_df["Fare"].median()).astype(int)
#Sex >>

dummy_col = pd.get_dummies(train_df['Sex'])

train_df = train_df.drop('Sex', axis=1).join(dummy_col)



dummy_col_test = pd.get_dummies(test_df['Sex'])

test_df = test_df.drop('Sex', axis=1).join(dummy_col_test)
# Ticket >> drop ticket



train_df = train_df.drop('Ticket',axis=1)

test_df = test_df.drop('Ticket',axis=1)
train_df.info()

test_df.info()
#Logistic Regression

lr = LogisticRegression()

lr.fit(train_df,Y_train)

Y_pred = lr.predict(test_df)

lr.score(train_df, Y_train)
#Random forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_df, Y_train)

Y_pred = random_forest.predict(test_df)

random_forest.score(train_df, Y_train)
result = pd.DataFrame({

        "PassengerId": titanic_test_df["PassengerId"],

        "Survived": Y_pred

    })

result.to_csv('titanic.csv', index=False)
result.info()