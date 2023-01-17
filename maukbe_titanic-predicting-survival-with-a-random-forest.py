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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df.shape
df.PassengerId.nunique()
# Passenger ID will be unique for every passenger

df.drop('PassengerId',inplace=True, axis=1)
df.head()
df.Ticket.nunique()
# There are 681 out of 891 unique ticket values so we may as well drop it

df.drop('Ticket', inplace=True,axis=1)
df.head()
df.Cabin.nunique()
df.Cabin.isna().sum()
# There's a lot of nans here - may as well ignore it

df.drop('Cabin', inplace=True, axis=1)
df.head()
# Someone's name shouldn't affect whether they survived or not

df.drop('Name', inplace=True, axis=1)
df.head()
# One hot encode the set so we can deal with the categorical data

df = pd.get_dummies(df, drop_first=True)
df.head()
# Let's fill the missing age values

df['Age'] = df['Age'].interpolate()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df.Survived, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, clf.predict(X_test))
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

sel.fit(X_train, y_train)
sel.get_support()
sel.get_params()
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                        criterion='gini', max_depth=None, max_features='auto',

                        max_leaf_nodes=None, max_samples=None,

                        min_impurity_decrease=0.0, min_impurity_split=None,

                        min_samples_leaf=1, min_samples_split=2,

                        min_weight_fraction_leaf=0.0, n_estimators=100,

                        n_jobs=None, oob_score=False, random_state=None,

                        verbose=0, warm_start=False)

clf.fit(X_train[['Pclass', 'Fare','Sex_male']], y_train)
accuracy_score(y_test, clf.predict(X_test[['Pclass', 'Fare','Sex_male']]))
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()
test_df = test_df[['Pclass', 'Sex', 'Fare']]
test_df.head()
test_df.isna().sum()
test_df = pd.get_dummies(test_df, drop_first=True)
# Fill in that one missing value

test_df['Fare'] = test_df['Fare'].interpolate()
predictions = clf.predict(test_df)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)