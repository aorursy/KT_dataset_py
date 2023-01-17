# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn as sk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# https://stackoverflow.com/questions/44009609/kaggle-typeerror-slice-indices-must-be-integers-or-none-or-have-an-index-me

def _revrt(X,m=None):

    """

    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.

    """

    if m is None:

        m = len(X)

    i = int(m // 2+1)

    y = X[:i] + np.r_[0,X[i:],0]*1j

    return np.fft.irfft(y)*m



from statsmodels.nonparametric import kdetools



# replace the implementation with new method.

kdetools.revrt = _revrt



# import seaborn AFTER replacing the method. 

import seaborn as sns

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.describe()
test.describe()
print(train.info())

print(test.info())
print(train.isnull().sum())

print('-------------')

print(test.isnull().sum())
plt.figure(figsize=[12,10])

sns.distplot(train.ix[train.Survived==1,'Age'].dropna().values, bins=range(0,81,1), hist=False,

            kde=True, rug=True, color='green')

sns.distplot(train.ix[train.Survived==0,'Age'].dropna().values, bins=range(0,81,1), hist=False,

            kde=True, rug=True, color='red')
plt.figure(figsize=[12,10])

sns.distplot(train.ix[train.Survived==1,'Fare'].dropna().values, bins=range(0,300,1), hist=False,

            kde=True, rug=True, color='green')

sns.distplot(train.ix[train.Survived==0,'Fare'].dropna().values, bins=range(0,300,1), hist=False,

            kde=True, rug=True, color='red')
ttrain = train.copy()

ttrain.ix[ttrain.Fare >= 200, 'Fare'] = 200

sns.boxplot(x='Sex', y='Fare', hue='Survived', data=ttrain)
sns.barplot('Embarked', 'Survived', data=train)
sns.barplot('Sex', 'Survived', data=train)
sns.barplot('Pclass', 'Survived', data=train)
sns.barplot('SibSp', 'Survived', data=train)
sns.barplot('Parch', 'Survived', data=train)
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf = clf.fit(train.ix[:, train.columns != 'Survived'], train.Survived)