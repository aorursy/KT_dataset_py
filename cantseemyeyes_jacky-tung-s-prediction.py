import seaborn
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import os

os.path.realpath(".")
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sns.boxplot(train['Age'])
sns.despine()
train.head()
train.describe()
train.columns
train[['Pclass','Survived']].head()
train.dtypes
pd.isna(train)
pd.isna(train).sum()
train[('Age')]
train.fillna(train.mean(), inplace=True)
print(train.isnull().sum())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
print(pd.__version__, np.__version__)
train_df = pd.read_csv('./data/train.csv', index_col='PassengerId')
test_df = pd.read_csv('./data/test.csv', index_col='PassengerId')

df = pd.concat([train_df, test_df], sort=True)
df.sample(10)
df[['Age', 'Sex']].isnull().sum()
df['Age'].describe()
max_age = df['Age'].max()
df['Age'].hist(bins=int(max_age))
df['decade'] = df['Age'].dropna().apply(lambda x: int(x/10))
df[['decade', 'Survived']].groupby('decade').mean().plot()
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
df['male'] = df['Sex'].map({'male': 1, 'female': 0})
df.sample(5)
df[['male','Survived']].groupby('male').mean()
train = df[df['Survived'].notnull()]

features = ['Age', 'male']
train_X = train[features]
train_y = train['Survived']
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(train_X, train_y)
classifier.score(train_X, train_y)
test = df[df['Survived'].isnull()]

test_X = test[features]
test_y = classifier.predict(test_X)
submit = pd.DataFrame(test_y.astype(int),
                      index=test_X.index,
                      columns=['Survived'])
submit.head()
submit.to_csv('prediction.csv')
