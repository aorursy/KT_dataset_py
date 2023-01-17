import pandas as pd

import numpy as np

import plotly

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
sns.boxplot(x='Survived', y='Age'

            , data=df)
sns.factorplot(x='Survived', hue='Sex', kind='count', data=df)
sns.factorplot(x='Survived', hue='Pclass', kind='count', data=df)
df.head()
sns.violinplot(x='Survived', y='Age',data=df)
mapping = {'female': 0, 'male':1}

df.replace({'Sex': mapping})
sns.factorplot(x='Survived', hue='Parch', kind='count', data=df)
sns.factorplot(x='Survived', hue='SibSp', kind='count', data=df)
df['Survived'].value_counts().plot(kind='pie')

plt.show()

df.columns
sns.factorplot(x='Survived', hue='Embarked', kind='count', data=df)
X_train = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

y_train = df['Survived']
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
X_train.Age = X_train.Age.fillna(round(X_train.Age.mean(), 0))
X_train.info()
X_train.replace({'Sex': mapping}, inplace=True)

X_train.head()
X_df = X_train.copy()

y_df = y_train.copy()
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y_df.values, test_size=0.2, random_state=42)
neighors = KNeighborsClassifier(n_neighbors=5)
neighors.fit(X_train, y_train)
pred = neighors.predict(X_test)
score = metrics.accuracy_score(y_test, pred)

print(score)
test_df = pd.read_csv('../input/test.csv')
test_df.info()
test_X_df = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
test_X_df.head()
test_X_df.replace({"Sex": mapping}, inplace=True)
test_X_df.head()
test_X_df.fillna(round(test_X_df.Age.mean(), 0), inplace=True)

test_X_df.head()
test_predict = neighors.predict(test_X_df.values)
submit_df = pd.DataFrame({

    'PassengerId': test_df['PassengerId'].values,

    'Survived': test_predict

})
submit_df.head()
submit_df.to_csv('submission.csv', index=False)