import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

import lightgbm as lgb

import numpy as np

import seaborn as sns

import os
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/chess/games.csv", delimiter=',')
df.head()
df.info()
df.columns
print(df.head())
df.isnull().sum()
df.describe()
from sklearn.preprocessing import LabelEncoder



categorical_column = ['rated', 'winner', 'victory_status']

                      



for i in categorical_column:

    le = LabelEncoder()

    df[i] = le.fit_transform(df[i])

print(df.head())
df[['black_rating', 'white_rating', 'turns', 'opening_ply', 'created_at', 'last_move_at']].hist(figsize=(12, 6), bins=50, grid=False)
sns.distplot(df['white_rating'])
sns.distplot(df['white_rating'],kde=False,bins=30)
sns.jointplot(x='white_rating',y='black_rating',data=df,kind='scatter')
sns.jointplot(x='white_rating',y='black_rating',data=df,kind='hex')
sns.jointplot(x='white_rating',y='black_rating',data=df,kind='reg')
sns.pairplot(df)
sns.pairplot(df,hue='winner',palette='coolwarm')
sns.barplot(x='winner',y='turns',data=df)
sns.barplot(x='winner',y='turns',data=df,estimator=np.std)
sns.countplot(x='winner',data=df)
sns.boxplot(x="winner", y="turns", data=df,palette='rainbow')
sns.violinplot(x="winner", y="turns", data=df,palette='rainbow')
df.corr()
sns.heatmap(df.corr())
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
g = sns.PairGrid(df)

g.map(plt.scatter)
sns.pairplot(df)
df['turns'].hist();
df['black_rating'].hist()
plt.style.use('ggplot')

df['white_rating'].hist()
plt.style.use('bmh')

df['white_rating'].hist()
plt.style.use('dark_background')

df['turns'].hist()
plt.style.use('ggplot')

df.plot.area(alpha=0.4)
plt.style.use('fivethirtyeight')

df.plot.area(alpha=0.4)