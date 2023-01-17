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
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier



# Importing data set

# Training set

train = pd.read_csv('/kaggle/input/titanic/train.csv')

# Test set

test = pd.read_csv('/kaggle/input/titanic/test.csv')

# The results

result = pd.DataFrame(test['PassengerId'])

# Preprocessing the data



# Getting info

# Training set

# Showing how many rows and coloumns in the table

train.shape

# Showing the first 5 rows in the table

train.head()

# Showing the null values

train.isnull().sum()
sns.heatmap(train.isnull())

plt.show()
# Test set

# Showing how many rows and columns in the table

test.shape

# Showing the first 5 rows in the table

test.head()

# Showing the null values

test.isnull().sum()


sns.heatmap(test.isnull())

plt.show()
# Getting rid of the columns we dont need

# Training set

# Deleting those columns

train.drop(['Name', 'PassengerId', 'Ticket',

            'Cabin', 'Fare'], axis=1, inplace=True)

train.shape
# Test set

# Deleting those columns



test.drop(['Name', 'PassengerId', 'Ticket',

           'Cabin', 'Fare'], axis=1, inplace=True)

test.shape

# Handling some of the missing values

# Training set

# Counting values

train['Embarked'].value_counts()

# Replacing the null values

train['Embarked'] = train['Embarked'].fillna(

    train['Embarked'].mode()[0]

)
# Showing the null values

train.isnull().sum()
# Handling the rest of the missing values

# Replace Age None Attribuate Values by Average age

train['Age'] = train['Age'].fillna(

    train['Age'].mean()

)

test['Age'] = test['Age'].fillna(

    test['Age'].mean()

)



train.isnull().sum() # Now no missing data
test.isnull().sum()
# Some Visualizations to analyze the data
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
# Explore Pclass vs Survived

g = sns.catplot(x="Pclass",y="Survived",data=train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")

# Encoding the categorical data

# Training set

train = pd.get_dummies(train)

test = pd.get_dummies(test)


# Show

train

test

# Now Time to build the model


target = pd.DataFrame(train['Survived'])

train.drop(['Survived'], axis=1, inplace=True)

# Building the machine Learning model

rfc = RandomForestClassifier(n_estimators=100,

                             max_features='auto',

                             criterion='entropy',

                             max_depth=10)
# Training the model

rfc.fit(train, target.values.ravel())
y_pred = rfc.predict(test)

temp = pd.DataFrame(y_pred)

result['Survived'] = y_pred

print(result)
result.to_csv("submission.csv", index=False)

print("Results converted to csv successfully")
# The accuracy is 77.751%