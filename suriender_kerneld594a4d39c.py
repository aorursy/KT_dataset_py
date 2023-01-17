# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('Data read')
df_train = pd.DataFrame()

df_train['Survived'] = train['Survived']

df_train['Pclass'] = train['Pclass']

#df_train['Age'] = train['Age']

df_train['SibSp'] = train['SibSp']

df_train['Parch'] = train['Parch']

#df_train['Fare'] = train['Fare']

df_train['Embarked'] = train['Embarked']
df_train.Embarked[df_train.Embarked == 'C'] = 1

df_train.Embarked[df_train.Embarked == 'S'] = 2

df_train.Embarked[df_train.Embarked == 'Q'] = 3



df_train['Sex'] = np.where(train['Sex'] == 'female', 1, 0)



test.Embarked[test.Embarked == 'C'] = 1

test.Embarked[test.Embarked == 'S'] = 2

test.Embarked[test.Embarked == 'Q'] = 3



#df_train = df_train.dropna(subset=['Age'])

df_train = df_train.dropna(subset=['Embarked'])



X_train = df_train.drop('Survived', axis=1)

Y_train = df_train.Survived

wanted_columns = X_train.columns
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

clf_gini.fit(X_train, Y_train)

test['Sex'] = np.where(test['Sex'] == 'female', 1, 0)



print(test.isnull().sum())

#test = test.dropna(subset = ['Fare'])

#test = test.dropna(subset = ['Age'])

predictions = clf_gini.predict(test[wanted_columns])
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

print(len(predictions))



submission.to_csv('submission.csv', index=False)
