# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info()

test.info()
survived_train = train.Survived

data = pd.concat([train.drop(['Survived'], axis=1), test])

data.info()
data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())

data['Embarked'] = data.Embarked.fillna('S')
# Check for missing data

for column in set(data.columns):

    print(column, data[column].isnull().sum())
data = pd.get_dummies(data, prefix = ['Embarked', 'Sex', 'Pclass'], columns = ['Embarked', 'Sex', 'Pclass'])
data['Cabin'] = train.Cabin.apply(lambda x: 1 if type(x) == str else 0)
# Combine SibSp and Parch to make new column which has family size

data['Fam_Size'] = data['SibSp'] + data['Parch'] + 1
data.info()
sns.boxplot(x = survived_train, y = 'Age', data = data)
sns.boxplot(x = survived_train, y = 'Fare', data = data)
# Drop columns



data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)

data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)

data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Age', 'Fare'], axis = 1)

data
from sklearn.model_selection import *

from sklearn.linear_model import *

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier



# Transform into binary variables

data_dum = data



# Split into test.train

data_train = data_dum.iloc[:891]

data_test = data_dum.iloc[891:]

# Create numpy arrays for our variables

# import numpy as np



X = np.array(data_train)

y = np.array(survived_train)



df_test_2 = np.array(data_test)



clf = DecisionTreeClassifier(max_depth=2)

clf.fit(X, y)



Y_pred = clf.predict(df_test_2)



test['Survived'] = Y_pred

# Create csv to submit to Kaggle:

# test[['PassengerId', 'Survived']].to_csv('../input/submit.csv', index=False)

test[['PassengerId', 'Survived']].to_csv('submit.csv', index = False)