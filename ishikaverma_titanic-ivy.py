# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()
test_df = test_df.drop(['Cabin'], axis=1)

for column in ['Sex', 'Embarked']:

  dummies = pd.get_dummies(test_df[column])

  test_df[dummies.columns] = dummies

test_df = test_df.drop(['Sex','Embarked'], axis=1)

test_df.head()
train_df['Survived'].value_counts()
train_df.shape
train_df.info()
train_df.isnull().sum()
sns.heatmap(train_df.isnull(), cbar=False)

from sklearn.preprocessing import Normalizer

train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

df = train_df.drop(['Cabin'], axis=1)



sns.heatmap(df.isnull(), cbar=False)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)

sns.heatmap(test_df.isnull(), cbar=False)
corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
for column in ['Sex', 'Embarked']:

  dummies = pd.get_dummies(df[column])

  df[dummies.columns] = dummies
df = df.drop(['Sex','Embarked'], axis=1)

df.head()
df['Survived'].value_counts().plot(kind='bar', title='Survival Rate');
df.plot(x="Survived", y=["female", "male"], kind ='hist',figsize=(10,5), stacked = True)
df.plot(x="Survived", y="Fare", kind ='hist',figsize=(10,5))
df.plot(x="Fare", y="Survived", kind = 'hist')
df1 = df.append(test_df, ignore_index=True)
df1 = df1.drop(['Name','Ticket'], axis=1)

df1["Survived"] = df1["Survived"].fillna(0)

df1.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1.drop('Survived',axis=1),df1['Survived'], test_size=0.30,random_state=101)
np.where(df1.values >= np.finfo(np.float64).max)
X_train.isnull().any(), y_train.isnull().any()
X_train.astype(np.float32).dtypes
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train.fillna(0).astype(np.float32), y_train.astype(np.float32))

predictions = logmodel.predict(X_test.fillna(0).astype(np.float32))
sns.heatmap(df1.isnull(), cbar=False)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
import sklearn.metrics as metrics

print("Accuracy:",metrics.accuracy_score(y_test, predictions))