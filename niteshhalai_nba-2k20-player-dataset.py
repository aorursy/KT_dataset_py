import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')

df
pd.DataFrame(df.columns)
df.info()
def a(salary):

    salary = salary.replace('$', '')

    return salary
df['salary'] = df['salary'].apply(a)

df['salary'] = df['salary'].astype('int64')
df
df.info()
pd.DataFrame(df['salary'].describe()).T
f, axes = plt.subplots(1,2,figsize=(15,5))



sns.violinplot(data=df, y='salary', ax = axes[0])

sns.distplot(df['salary'], ax = axes[1])

plt.show()
pd.DataFrame(df['position'].value_counts()).T

mean_by_position = pd.DataFrame(df.groupby(by='position').mean()['salary'].sort_values(ascending=False))

median_by_position = pd.DataFrame(df.groupby(by='position').median()['salary'].sort_values(ascending=False))

count_by_position = pd.DataFrame(df.groupby(by='position').count()['salary'].sort_values(ascending=False))

salary_by_position = mean_by_position.merge(how='outer', left_index=True, right_index=True, right=median_by_position['salary'])

salary_by_position = salary_by_position.merge(how='outer', left_index=True, right_index=True, right=count_by_position['salary'])

salary_by_position.rename({'salary_x': 'mean', 'salary_y': 'median', 'salary': 'count'}, axis='columns', inplace=True)

salary_by_position
salary_by_position.plot(figsize=(15,5))

plt.show()
f, axes = plt.subplots(1,1,figsize=(15,5))

sns.countplot(data=df, x='position')

plt.show()
f, axes = plt.subplots(1,1,figsize=(15,5))



sns.violinplot(data=df, x='position', y='salary')

plt.show()