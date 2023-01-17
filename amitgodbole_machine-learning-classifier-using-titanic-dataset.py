# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Print Train Sample Data

df_train.head()
# Print Test Sample Data

df_test.head()
df_train.info()
df_train.describe()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.5, aspect=2.0)

grid.map(plt.hist, 'Age', alpha=.5, bins=25)

grid.add_legend();
df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



df_test['Embarked'] = df_test['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())

df_train.loc[ df_train['Fare'] <= 7.91, 'Fare'] 						         = 0

df_train.loc[(df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454), 'Fare']   = 1

df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31), 'Fare']     = 2

df_train.loc[ df_train['Fare'] > 31, 'Fare'] 							         = 3

df_train['Fare'] = df_train['Fare'].astype(int)



df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

df_test.loc[ df_test['Fare'] <= 7.91, 'Fare'] 						         = 0

df_test.loc[(df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.454), 'Fare']   = 1

df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31), 'Fare']     = 2

df_test.loc[ df_test['Fare'] > 31, 'Fare'] 							         = 3

df_test['Fare'] = df_test['Fare'].astype(int)
age_avg 	   = df_train['Age'].mean()

df_train['Age'][np.isnan(df_train['Age'])] = age_avg



df_train.loc[ df_train['Age'] <= 16, 'Age'] 					      = 0

df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1

df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2

df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3

df_train.loc[ df_train['Age'] > 64, 'Age']                            = 4



df_train['Age'] = df_train['Age'].astype(int)



age_avg 	   = df_test['Age'].mean()

df_test['Age'][np.isnan(df_test['Age'])] = age_avg



df_test.loc[ df_test['Age'] <= 16, 'Age'] 					       = 0

df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1

df_test.loc[(df_test['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2

df_test.loc[(df_test['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3

df_test.loc[ df_test['Age'] > 64, 'Age']                           = 4



df_test['Age'] = df_test['Age'].astype(int)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch']

df_train = df_train.drop(drop_elements, axis = 1)

df_test = df_test.drop(drop_elements, axis = 1)
df_train.head()
df_test.head()
df_train_data = df_train.drop("Survived", axis=1)

df_train_output = df_train["Survived"]
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()

logreg.fit(df_train_data, df_train_output)

acc_log = round(logreg.score(df_train_data, df_train_output) * 100, 2)

acc_log
svc = SVC()

svc.fit(df_train_data, df_train_output)

acc_svc = round(svc.score(df_train_data, df_train_output) * 100, 2)

acc_svc
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(df_train_data, df_train_output)

Y_pred = random_forest.predict(df_test)

random_forest.score(df_train_data, df_train_output)

acc_random_forest = round(random_forest.score(df_train_data, df_train_output) * 100, 2)

acc_random_forest
test_df = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)