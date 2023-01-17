from sys import version as py_version

from scipy import __version__ as sp_version

from numpy import __version__ as np_version

from pandas import __version__ as pd_version

from matplotlib import __version__ as mplib_version

from sklearn import __version__ as skl_version



print('python: {}'.format(py_version))

print('scipy: {}'.format(sp_version))

print('numpy: {}'.format(np_version))

print('pandas: {}'.format(pd_version))

print('matplotlib: {}'.format(mplib_version))

print('sklearn: {}'.format(skl_version))
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
print(check_output(["ls", "-l","."]).decode("utf8"))
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
dataset = pd.read_csv('../input/train.csv')
dataset.head()
dataset.shape
dataset.dtypes
dataset.describe()
dataset.describe(include=['object'])
dataset.groupby('Survived').size()
dataset.groupby('Pclass').size()
dataset.groupby('Sex').size()
dataset.groupby('Embarked').size()
dataset.groupby(['Survived', 'Pclass']).size().unstack()
dataset.groupby(['Survived', 'Sex']).size().unstack()
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False, layout=(2,4), figsize=(10,6))

pyplot.show()
dataset.hist(figsize=(10,6), sharex=False, sharey=False)

pyplot.show()

scatter_matrix(dataset, figsize=(8,8))

pyplot.show()
dataset.corr()
#dataset['Sex'] = dataset['Sex'].astype('category')
#pd.get_dummies(dataset['Sex'], drop_first=True).head()
#dataset = pd.concat([dataset, pd.get_dummies(dataset['Sex'], drop_first=True)], axis=1)
names = ['PassengerID','Survived','Pclass','Age', 'SibSp','Parch','Fare']

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')

fig.colorbar(cax)

ticks = np.arange(0,7,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

fig.set_size_inches(10,10)

pyplot.show() 
train = dataset.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

train.head()
print(train.shape)

print(train.dtypes)
X_train = dataset.drop(['Survived', 'PassengerId','Name','Ticket','Cabin'], axis=1)

Y_train = dataset['Survived']

X_train.shape, Y_train.shape
X_train['Sex'] = dataset['Sex'].astype('category').cat.codes

X_train['Embarked'] = dataset['Embarked'].astype('category').cat.codes

X_train.dtypes
X_train.describe()
X_train['Age'].fillna(X_train['Age'].median(), inplace=True)
X_train.describe()
cart = DecisionTreeClassifier()

cart.fit(X_train, Y_train)
test = pd.read_csv('../input/test.csv')
print(test.shape)

print(test.dtypes)
test.head()
X_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

X_test['Sex'] = test['Sex'].astype('category').cat.codes

X_test['Embarked'] = test['Embarked'].astype('category').cat.codes

X_test.dtypes
X_test.describe()
X_test['Age'].fillna(X_test['Age'].median(), inplace=True)

X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True)
X_test.describe()
predictions = cart.predict(X_test)
predictions
submission = pd.DataFrame({

    'PassengerId': test['PassengerId'],

    'Survived': predictions

})
submission.head()
submission.to_csv('submission.csv', index=False)