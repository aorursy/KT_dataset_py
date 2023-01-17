# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
for i in range(len(df.columns)):
    percentage = df[df.columns[i]].isnull().sum()/df[df.columns[i]].shape[0]
    if percentage > 0:
        print('There is {:.2%} missing data in column {}'.format(percentage, df.columns[i]))
    else:
        pass
df['Year'] = df['Year'].fillna(df['Year'].mode)
df['Publisher'] = df['Publisher'].fillna(max(df['Publisher'].value_counts().values))
df.isnull().sum()
df_cleaned = df.copy()
NA = df_cleaned[['NA_Sales', 'Genre']]
EU = df_cleaned[['EU_Sales', 'Genre']]
JP = df_cleaned[['JP_Sales', 'Genre']]
f, axs = plt.subplots(3,1, figsize=(15,6))
sns.barplot(x='Genre', y='NA_Sales', data=df_cleaned, ax=axs[0], ci=None)
sns.barplot(x='Genre', y='EU_Sales', data=df_cleaned, ax=axs[1], ci=None)
sns.barplot(x='Genre', y='JP_Sales', data=df_cleaned, ax=axs[2], ci=None)
def top_n_publishers(n):
    publishers = df_cleaned.groupby('Publisher')['Global_Sales'].sum().reset_index().sort_values(by='Global_Sales', ascending=False)
    p = publishers.iloc[:n,:]
    fig, ax = plt.subplots(figsize=(9,6))
    sns.barplot(x='Publisher', y='Global_Sales', data=p, ci=None, orient='v', color='blue')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=10)
    plt.title('Top {} Publishers by Sales'.format(n), fontsize=15, fontweight='bold')
    plt.xlabel('Companies')
    plt.ylabel('Sales')
    plt.show()
top_n_publishers(15)
df_cleaned['Year'] = df_cleaned['Year'].astype(str)
time = df_cleaned.groupby('Year')['Global_Sales'].sum().reset_index()[:-1]
time['Year'] = time['Year'].astype(float)
time['Year'] = time['Year'].astype(int)
f, ax = plt.subplots(figsize=(20,6))
x = sns.lineplot(x='Year', y='Global_Sales', data=time)
plt.title('Video game sales 1980 - 2020', fontsize=20, fontweight='bold')
plt.xlabel('Year',fontsize=15)
plt.ylabel('Global Sales', fontsize=15)