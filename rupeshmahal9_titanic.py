# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

df1 = pd.read_csv('../input/gender_submission.csv')

combine=[train_df,test_df]
train_df.head(6)
train_df.dtypes
np.where(pd.isnull(train_df))
train_df['Cabin'] = train_df.fillna(train_df['Cabin'].value_counts().index[0])

train_df.head()
train_df[['Age']] = train_df[["Age"]].apply(pd.to_numeric)
train_df.head(10)
import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import seaborn as sns
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)


corr = train_df.corr()

corr.style.background_gradient(cmap='coolwarm')
train_df.head()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

train_df = train_df.drop(['Name'], axis=1)
bins = [0, 18,  60, np.inf]

names = [0,1,2]

train_df['Agecat'] = pd.cut(train_df['Age'], bins, labels=names)
test_df.head()
test_df['Agecat'] = pd.cut(test_df['Age'], bins, labels=names)
test_df.head()
test_df = test_df.drop(['Age'], axis=1)

train_df = train_df.drop(['Age'], axis=1)
corr = train_df.corr()

corr.style.background_gradient(cmap='coolwarm')
combine = [train_df, test_df]
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
freq_port = train_df.Fare.dropna().mode()[0]

for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(freq_port)

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#train_df.dropna(subset=['Agecat'],axis=0,inplace=True)



train_df.head(100)
freq_port = train_df.Agecat.dropna().mode()[0]

for dataset in combine:

    dataset['Agecat'] = dataset['Agecat'].fillna(freq_port)
#np.where(pd.isnull(test_df))

np.where(pd.isnull(train_df))
train_df=train_df.drop("PassengerId", axis=1)

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
sub=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred})

sub.to_csv("submit.csv",index=False)
sub.head()