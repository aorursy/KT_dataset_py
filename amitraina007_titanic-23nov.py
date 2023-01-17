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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train.describe()
train.info()
import seaborn as sns
sns.boxplot(data=train, x='Age')
train['Survived'].value_counts()
train['Pclass'].value_counts()
sns.pairplot(x_vars='Pclass', y_vars='Survived', data=train)
#lets use Pclass and Age as x variables
sns.pairplot(x_vars='Age', y_vars='Survived', data=train)
#nulls in Age; so we need to treat them first
train['Age'].describe()
train.fillna(value = {'Age':np.mean(train['Age'])}, inplace=True)
train.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'Age']], train['Survived'], test_size = 0.2, random_state = 1)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier
dt1 = DecisionTreeClassifier()
dt1.fit(X=X_train, y=y_train)
dt1.score(X=X_train, y=y_train)
#test accuracy

dt1.score(X=X_test, y=y_test)
import graphviz
from sklearn.tree import export_graphviz

import IPython

from IPython.display import display
tree_visual2 =export_graphviz(dt1, out_file=None, feature_names=['Pclass', 'Age'], filled=True,

                      special_characters=True, rotate=True, precision=3)

IPython.display.display(graphviz.Source(tree_visual2))
dt2 = DecisionTreeClassifier(max_depth=2)
dt2.fit(X=X_train, y = y_train)

dt2.score(X=X_train, y = y_train)
tree_visual2 =export_graphviz(dt2, out_file=None, feature_names=['Pclass', 'Age'], filled=True,

                      special_characters=True, rotate=True, precision=3)

IPython.display.display(graphviz.Source(tree_visual2))