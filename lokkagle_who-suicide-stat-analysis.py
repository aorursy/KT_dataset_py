# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')
data.head()
data.info()
data.isna().sum()
# data distribution for suicide_no and population, cause we need to impute with mean() or median()
data.select_dtypes(include= np.float64).hist(figsize = (15,5))
plt.show()
data.select_dtypes(include= np.float64).describe().T
print(data['suicides_no'].mean())
print(data['suicides_no'].median())
for i in data.select_dtypes(include= np.float64).columns:
    data[i] = data[i].replace(np.NaN, data[i].median())
data.isna().sum()
data.head()
# check for the columns which has below 10 classes 
for column in list(data.columns[:-2]):
    if len(data[column].unique()) <= 10:
        print("{}: {}".format(column ,data[column].unique()))
plt.style.use('ggplot')
sns.relplot(x= 'suicides_no', y= 'population', data = data, col= 'sex')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x= 'suicides_no', y= 'population', data = data, hue = 'sex')
plt.show()
plt.style.use('ggplot')
sns.relplot(x= 'suicides_no', y= 'population', data = data, col= 'age')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x= 'suicides_no', y= 'population', data = data, hue = 'age')
plt.show()
plt.figure(figsize= (15,5))
sns.regplot(x= 'suicides_no', y= 'population', data = data)
plt.show()
# mean population by sex and age
data.groupby(['sex', 'age'])['population'].mean()
data.groupby(['sex', 'age'])['suicides_no'].mean()
data.groupby(['sex', 'age'])['suicides_no'].mean().plot(figsize = (15,5))
plt.tight_layout()
plt.show()
data.groupby(['country'])['population'].mean()[:10].plot(kind = 'line', figsize = (15,5))
plt.tight_layout()
plt.show()
# top 20 countries suicides on average
data.groupby(['country'])['suicides_no'].mean().sort_values(ascending = False)[:20]
# .plot(kind = 'line', figsize = (15,5))
# plt.tight_layout()
# plt.show()
# top 20 countries suicides on average
data.groupby(['country'])['suicides_no'].mean().sort_values(ascending = False)[:20].plot(kind = 'barh', figsize = (15,5))
plt.tight_layout()
plt.show()
data.groupby(['year'])['suicides_no'].mean().plot(kind = 'line', figsize = (15,5))
plt.show()
plt.figure(figsize= (15,5))
sns.lineplot(x = 'year', y = 'suicides_no', data = data, hue = 'sex')
plt.show()
plt.figure(figsize= (15,5))
sns.lineplot(x = 'year', y = 'suicides_no', data = data, hue = 'age')
plt.show()
data.groupby('country')['suicides_no'].max().sort_values(ascending = False)[:20]
data.groupby(['year','country'])['suicides_no'].max().sort_values(ascending = False)
data.groupby('country')['suicides_no'].min().sort_values()[:20]
