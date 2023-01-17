# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.read_csv('foo.csv')

train_df = pd.read_csv('../input/train.csv')

train_df.head()

test_df = pd.read_csv('../input/test.csv')

test_df.head()
train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].info()

test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].info()

mode_age = test_df['Age'].mode()[0]

test_df['Age'] = test_df['Age'].fillna(value=mode_age)





median_fare = test_df['Fare'].median()

test_df['Fare'] = test_df['Fare'].fillna(value=median_fare)

test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].info()
train_df = train_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna(how='any')

train_df.info()
clf = svm.SVC()



clf.fit(train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], train_df['Survived'])

result = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived':clf.predict(test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

})

result.to_csv('result.csv')