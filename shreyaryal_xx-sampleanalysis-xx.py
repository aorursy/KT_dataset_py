#setup...

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
g = sns.FacetGrid(df, col='Sex', row = 'Survived', hue='Pclass')

g.map(plt.hist, 'Pclass')


df_test = pd.read_csv('../input/test.csv')

df_test['is_male'] = pd.get_dummies(df_test['Sex'])['male']
model = LogisticRegression();

model_params = ['is_male', 'Pclass']

target_params = 'Survived'

X = df[model_params]

y = df[target_params]

model.fit(X,y)

model.score(X,y)
X_test = df_test[model_params]

df_test['Survived'] = model.predict(X_test)
df_test

df[['PassengerId','Survived']].to_csv('predict.csv')