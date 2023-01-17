# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")



# preview the data

train.head()
test.head()
train.shape
test.shape
train.isnull().sum()
import seaborn as sns

sns.heatmap(train.isnull())
test.isnull().sum()
sns.heatmap(test.isnull())
sns.countplot(x='Survived', data=train, hue='Pclass')
sns.countplot(x='SibSp', data=train, hue='Survived')
sns.boxplot(x='Pclass', y='Age', data=train)
train.groupby('Pclass')['Age'].max()
train.info()
train['Age'].median()
impute_value = train['Age'].median()

train['Age'] = train['Age'].fillna(impute_value)

test['Age'] = test['Age'].fillna(impute_value)
sns.heatmap(train.isnull())
train['IsFemale'] = (train['Sex'] == 'female').astype(int)

test['IsFemale'] = (test['Sex'] == 'female').astype(int)
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
X_train[:5]
y_train[:5]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict[:50]
type(y_predict)
a = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": y_predict

})
a.to_csv("mysubmission.csv", index=False)