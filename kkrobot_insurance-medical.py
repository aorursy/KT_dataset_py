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
df = pd.read_csv('/kaggle/input/datasets_13720_18513_insurance.csv')

df.shape
df.describe()
df.head()
df[pd.isnull(df)]
df.isnull().sum()
df.dtypes
df.columns
%matplotlib notebook
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
for style in plt.style.available:
    print(style)
df.age.unique()
df.charges.unique()
plt.plot(df.bmi)
plt.figure(figsize=(200,40))
df.age.hist()
df.charges.hist()
df.age.hist()
df.region.hist()
sns.set(style='dark')
f, ax = plt.subplots(1,1, figsize=(12, 8))
ax = sns.distplot(df['age'], kde = True, color = 'r')
plt.title('Various age groups')
plt.figure(figsize=(46,20))

plt.title("Average age , by children")

sns.barplot(x=df.age, y=df.children)

plt.ylabel(" Average Children")
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='pastel')
f, ax = plt.subplots(1, 1, figsize=(20, 18))
ax = sns.barplot(x='region', y='charges', hue='children', data=df, palette='rocket_r')
sns.set(style='darkgrid')
charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)
f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.barplot(charges.head(), charges.head().index, palette='Blues')
f, ax = plt.subplots(1,1, figsize=(12,8))
ax = sns.barplot(x = 'region', y = 'charges',
                 hue='smoker', data=df, palette='viridis')
f, ax = plt.subplots(1,1, figsize=(12,8))
ax = sns.barplot(x = 'region', y = 'age',
                 hue='smoker', data=df, palette='Accent')
df.boxplot(column = ['age','bmi','charges'])
df[['age','charges']]
ax=sns.lmplot(x='age' , y='charges' , data=df , hue='smoker' ,height=7, palette='gist_heat' )
ax=sns.lmplot(x='bmi'  , y ='charges' ,data = df , hue='smoker' ,height=8, palette='gist_rainbow_r')
ax=sns.lmplot(x='children' , y='charges', data=df , hue='smoker' , height=5,palette= 'gist_stern')

ax=sns.lmplot(x="bmi" , y="charges" , data=df , hue='sex' , height =9,  palette='gist_heat_r')

ax=sns.lmplot(x="age" , y="charges" , data=df , hue='sex' , height =12,  palette='Set1')

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.boxplot(x = 'children', y = 'charges', data=df,orient='v', hue='smoker', palette='gray')
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.boxplot(x = 'sex', y = 'charges', data=df,orient='v', hue='region', palette='gray')
f, ax = plt.subplots(1, 1, figsize=(16,8))
ax = sns.heatmap(df.corr(), annot=True, cmap='Greens')
sns.heatmap(df.corr())
