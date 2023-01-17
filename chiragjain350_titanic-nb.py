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
test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")

train_df.head()
import seaborn as sns

sns.heatmap(train_df.corr())
train_df.corr()['Survived']
train_df.isnull().sum()
train_df.info()
sns.boxplot(x='Pclass',y='Age',data=train_df)
train_df['Age'].hist()
train_df['Age'] = train_df['Age'].replace(to_replace = np.nan , value = train_df['Age'].mean())
train_df.isnull().sum()
train_df.drop('Cabin',axis=1,inplace = True)
train_df.head()
train_df.info()
train_df.dropna(inplace=True)
train_df.info()
sex = pd.get_dummies(train_df['Sex'],drop_first=True)

embark = pd.get_dummies(train_df['Embarked'],drop_first=True)
train_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train_df = pd.concat([train_df,sex,embark],axis=1)
train_df.head()
test_df.isnull().sum()
test_df.head()
test_df['Age'] = test_df['Age'].replace(to_replace = np.nan , value = test_df['Age'].mean())
test_df.drop('Cabin',axis=1,inplace=True)
test_df.dropna(inplace=True)
test_df.isnull().sum()
test_df.info()
sex = pd.get_dummies(test_df['Sex'],drop_first=True)

embark = pd.get_dummies(test_df['Embarked'],drop_first=True)

test_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test_df = pd.concat([test_df,sex,embark],axis=1)
test_df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = train_df.drop('Survived',axis=1)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
logmodel = LogisticRegression()

logmodel.fit(X_train,train_df['Survived'])
predictions = logmodel.predict(test_df)
predictions
test_df['Survived'] = predictions
test_df.head()