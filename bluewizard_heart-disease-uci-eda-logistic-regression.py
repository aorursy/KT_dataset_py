import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.target = df.target.replace({0:1, 1:0})
df.target.value_counts()
bins = ['sex', 'fbs', 'exang']

cats = ['cp', 'restecg', 'slope', 'thal']

ords = ['ca']

nums = ['age', 'oldpeak', 'trestbps', 'chol', 'thalach']

target = ['target']
df.cp = df.cp.replace({0:'Asympt.', 1:'Atypical', 2:'Non', 3:'Typical'})

df.restecg = df.restecg.replace({0:'LV hyper', 1:'Normal', 2:'ST-T wave'})

df.slope = df.slope.replace({0:'down', 1:'up', 2:'flat'})

df.thal = df.thal.replace({0:'NA', 1:'Fixed', 2:'Normal', 3:'Revers.'})
X_train, X_test, y_train, y_test = train_test_split(df, 

                                                    df.target, 

                                                    test_size = 0.2, 

                                                    random_state = 42,

                                                    stratify = df.target)
X_train[X_train.target == 0].drop(cats + bins + target, 

                                  axis=1).describe().loc[['mean', 'std']]
X_train[X_train.target == 1].drop(cats + bins + target, 

                                  axis=1).describe().loc[['mean', 'std']]
fig = plt.figure(figsize=(8, 6))

fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.01, top=0.95)



for i, var in enumerate(cats):

    i = i + 1

    ax = fig.add_subplot(2, 2, i)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    sns.countplot(data = X_train, x = var, hue = 'target', ax = ax)



plt.show()
df.restecg = df.restecg.replace({'Normal':0, 'LV hyper':1, 'ST-T wave':1})

df.thal = df.thal.replace({'NA':0, 'Normal':0, 'Fixed': 1, 'Revers.': 1})

X_train, X_test, y_train, y_test = train_test_split(df, 

                                                    df.target, 

                                                    test_size = 0.2, 

                                                    random_state = 42,

                                                    stratify = df.target)
bins = ['sex', 'fbs', 'exang', 'thal', 'restecg']

cats = ['cp', 'slope']
fig = plt.figure(figsize=(8, 6))

fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.01, top=0.95)



for i, var in enumerate(bins):

    i = i + 1

    ax = fig.add_subplot(2, 3, i)

    sns.countplot(data = X_train, x = var, hue = 'target', ax = ax)



plt.show()
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_transformer

clt = make_column_transformer(

    (StandardScaler(), nums),

    (OneHotEncoder(), cats)

)



clt.fit(X_train)

X_train_transformed = clt.transform(X_train)

X_test_transformed = clt.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_transformed, y_train)
lr.score(X_test_transformed, y_test)
from sklearn.metrics import confusion_matrix

y_pred = lr.predict(X_test_transformed)

confusion_matrix = confusion_matrix(y_test, y_pred)

confusion_matrix