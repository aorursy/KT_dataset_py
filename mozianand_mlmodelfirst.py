# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in # Store target variable of training data in a safe plac



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Figures inline and set visualization style

%matplotlib inline

sns.set()



# Import data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Store target variable of training data in a safe place

survived_train = df_train.Survived



# Concatenate training and test sets to create a new Merged_data

Merged_data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])



# Check info Merged_data 

Merged_data.info()
# Fill the missing numerical variables

Merged_data['Age'] = Merged_data.Age.fillna(Merged_data.Age.median())

Merged_data['Fare'] = Merged_data.Fare.fillna(Merged_data.Fare.median())



# Check out info of data

Merged_data.info()
Merged_data = pd.get_dummies(Merged_data, columns=['Sex'], drop_first=True)

Merged_data.head()
# Select columns and view head

Merged_data = Merged_data[['Sex_male', 'Fare', 'Age','Pclass','SibSp']]

Merged_data.head()
data_train = Merged_data.iloc[:891]

data_test = Merged_data.iloc[891:]
X = data_train.values

test = data_test.values

y = survived_train.values
# Instantiate model and fit to data

clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X, y)
# Make predictions and store in 'Survived' column of df_test

Y_pred = clf.predict(test)

df_test['Survived'] = Y_pred
#df_test[['PassengerId', 'Survived']].to_csv('2nd_dec_tree_version.csv', index=False)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=5000)

rfc.fit(X, y)
rfc_pred =rfc.predict(test)

df_test['Survived'] = rfc_pred

df_test[['PassengerId', 'Survived']].to_csv('Randm_Forest_versionMD.csv', index=False)
