# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Loading Train and Test Set

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
train_df.describe()
#Droping Ticket and Cabin feature because of High Cardinality

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)



train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)



combine = [train_df, test_df]



train_df.tail()
#Converting Categorial Feature in Train and Test Datasets



from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

train_df['Sex'] = le.fit_transform(train_df['Sex']) 

test_df['Sex']  = le.fit_transform(test_df['Sex']) 

#Handling NAN in Embarked column - Updating Train and Test

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])

    dataset['Embarked'] = dataset['Embarked'].fillna(test_df.Embarked.dropna().mode()[0])





# OneHotEncoding - Creating 3 Features Embarked_0, Embarked_1 and Embarked_2

from sklearn.preprocessing import OneHotEncoder

embarked_ohe = OneHotEncoder()



X_TRAIN = embarked_ohe.fit_transform(train_df.Embarked.values.reshape(-1,1)).toarray()

X_TEST = embarked_ohe.fit_transform(test_df.Embarked.values.reshape(-1,1)).toarray()



dfOneHot = pd.DataFrame(X_TRAIN, columns = ["Embarked_"+str(int(i)) for i in range(X_TRAIN.shape[1])])

dfOneHot_TEST = pd.DataFrame(X_TEST, columns = ["Embarked_"+str(int(i)) for i in range(X_TEST.shape[1])])

train_df = pd.concat([train_df, dfOneHot], axis=1)

test_df = pd.concat([test_df, dfOneHot_TEST], axis=1)



train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)



test_df.head()
#Adding a new feature to add SibSp and Parch 

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1



test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#Handling Age NAN - 

train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)

train_df.head()



test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)

test_df.head()
#Checking Survived and Embarked_1._0._2 (New Features)

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Embarked_1', bins=20)
#Hnadling NAN in Fare

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
# MODEL

X_train = train_df.drop("Survived", axis=1) # Removing the dependent feature - Y

Y_train = train_df["Survived"] 

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_test  = test_df.copy()

X_train.shape, Y_train.shape, X_test.shape

# machine learning - Logisctic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#Machine Learning using RandomForestClassifier



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

acc_rf = round(rf.score(X_train, Y_train) * 100, 2)

acc_rf

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })





submission.to_csv('/kaggle/working/praveenk_submission2.csv', index=False)