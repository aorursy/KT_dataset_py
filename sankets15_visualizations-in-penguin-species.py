import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

df.head()
sns.pairplot(df, hue='species')
plt.scatter(df['culmen_length_mm'], df['flipper_length_mm'])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

df.isnull().sum()
for col in list(df.columns):

    if df[col].dtypes=='object':

        print('Levels of',col,':',df[col].value_counts().keys())
df['sex'].replace('.',np.nan, inplace=True)

df.isnull().sum()
for col in list(df.columns):

    if df[col].dtypes=='object':

#         print('Max (mode) of',col,'****',type(df[col].mode()[0]))

        df[col].fillna(df[col].mode()[0], inplace=True)

    else:

        df[col].fillna(df[col].median(), inplace=True)

df.isnull().sum()
sns.pairplot(df, hue='species')
fig, ax = plt.subplots(figsize=(12,6))

sns.heatmap(df.corr(), annot=True, ax=ax)
fig, ax = plt.subplots(figsize =(8,5))

ax.hist(df[df.columns[2]], rwidth=0.8, alpha=0.55)

ax.axvline(df[df.columns[2]].median(), color='red')

ax.axvline(df[df.columns[2]].mean())