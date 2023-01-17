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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
plt.figure(figsize=[10,10])
sns.distplot(train['Age'].dropna().values, bins=range(0,17), kde=False, color="#007598")
sns.distplot(train['Age'].dropna().values, bins=range(16, 30), kde=False, color="#7B97A0")
sns.distplot(train['Age'].dropna().values, bins=range(29, 50), kde=False, color="#06319B")
sns.distplot(train['Age'].dropna().values, bins=range(49,65), kde=False, color="#007598")
sns.distplot(train['Age'].dropna().values, bins=range(64,81), kde=False, color="#000000",
            axlabel='Age')
plt.show()
train['Age_Category'] = pd.cut(train['Age'],
                        bins=[0,16,29,49,64,81])
plt.subplots(figsize=(10,10))
sns.countplot('Age_Category',hue='Survived',data=train, palette='RdBu_r')
plt.show()
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 29), 'Age'] = 1
train.loc[(train['Age'] > 29) & (train['Age'] <= 49), 'Age'] = 2
train.loc[(train['Age'] > 49) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
    
train.head()
train['Family'] = train['SibSp'] + train['Parch'] + 1
train['Alone'] = 0
train.loc[train['Family'] == 1, 'Alone'] = 1
train['Sex'].replace("male", 0, inplace=True)
train['Sex'].replace("female", 1, inplace=True)
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
training, testing = train_test_split(train, test_size=0.2, random_state=0)
cols = ['Sex', 'Age', 'Pclass', 'Family', 'Fare', 'Embarked']
tcols = np.append(['Survived'],cols)
df = training.loc[:,tcols].dropna()

X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
y_red_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X, y)*100, 2)
acc_random_forest
