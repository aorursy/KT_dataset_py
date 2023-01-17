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

import matplotlib.pyplot as plt

import scipy.stats as st

%matplotlib inline



sns.set(style="whitegrid")
# ignore warnings



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
# print the shape

print('The shape of the dataset : ', df.shape)
# preview dataset

df.head()
# summary of dataset

df.info()
df.dtypes
# statistical properties of dataset

df.describe()
df.columns
df['target'].nunique()
df['target'].unique()
df['target'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="target", data=df)

plt.show()
df.groupby('sex')['target'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="sex", hue="target", data=df)

plt.show()
ax = sns.catplot(x="target", col="sex", data=df, kind="count", height=5, aspect=1)
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(y="target", hue="sex", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="target", data=df, palette="Set3")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="target", data=df, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3))

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="target", hue="fbs", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="target", hue="exang", data=df)

plt.show()
correlation = df.corr()
correlation['target'].sort_values(ascending=False)
df['cp'].nunique()
df['cp'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="cp", data=df)

plt.show()
df.groupby('cp')['target'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="cp", hue="target", data=df)

plt.show()
ax = sns.catplot(x="target", col="cp", data=df, kind="count", height=8, aspect=1)
df['thalach'].nunique()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

x = pd.Series(x, name="thalach variable")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

ax = sns.distplot(x, bins=10, vertical=True)

plt.show()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

x = pd.Series(x, name="thalach variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

x = pd.Series(x, name="thalach variable")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(10,6))

x = df['thalach']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="target", y="thalach", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="target", y="thalach", data=df, jitter = 0.01)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="target", y="thalach", data=df)

plt.show()
plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Heart Disease Dataset')

a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')

a.set_xticklabels(a.get_xticklabels(), rotation=90)

a.set_yticklabels(a.get_yticklabels(), rotation=30)           

plt.show()
num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]

sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')

plt.show()

df['age'].nunique()
df['age'].describe()
f, ax = plt.subplots(figsize=(10,6))

x = df['age']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="target", y="age", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="target", y="age", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.scatterplot(x="age", y="trestbps", data=df)

plt.show()

f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="age", y="trestbps", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.scatterplot(x="age", y="chol", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="age", y="chol", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.scatterplot(x="chol", y = "thalach", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="chol", y="thalach", data=df)

plt.show()
# check for missing values



df.isnull().sum()
#assert that there are no missing values in the dataframe



assert pd.notnull(df).all().all()

#assert all values are greater than or equal to 0



assert (df >= 0).all().all()

df['age'].describe()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=df["age"])

plt.show()
df['trestbps'].describe()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=df["trestbps"])

plt.show()

df['chol'].describe()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=df["chol"])

plt.show()

df['thalach'].describe()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=df["thalach"])

plt.show()
df['oldpeak'].describe()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=df["oldpeak"])

plt.show()
