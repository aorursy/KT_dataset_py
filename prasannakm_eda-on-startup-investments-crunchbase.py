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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
%matplotlib inline
import seaborn as sns
df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = "ISO-8859-1")
df.head()
df.shape
fig = plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
df.isnull().sum()
#Removing the rows with NULL values
startups = df.dropna(how='all')
startups.shape
startups.tail()
startups.columns
startups.columns = startups.columns.str.strip()
startups.columns
startups['market'].value_counts()[:20]
fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['market'],order=startups['market'].value_counts()[:20].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
def clear_str(x):
    x = x.replace(',','').strip()
    x = x.replace('-','0')
    return int(x)
startups['funding_total_usd'] = startups['funding_total_usd'].apply(lambda x: clear_str(x))
startups['funding_total_usd'].isnull().sum()
fig = plt.subplots(figsize=(20,10))
market_fund = startups.groupby('market').sum()['funding_total_usd'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = startups,x = market_fund.index, y= market_fund.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['country_code'],order=startups['country_code'].value_counts()[:15].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['status'],order=startups['status'].value_counts()[:15].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
plt.figure(figsize = (10,10))
startups.status.value_counts().plot(kind='pie', explode=(0, 0.05, 0.1),autopct='%1.1f%%',startangle=45)
plt.title('Status')
plt.show()
startups.columns
sns.factorplot('name',data=startups[(startups['name']=='Facebook')|(startups['name']=='Alibaba')|(startups['name']=='Uber')],kind='count',hue='funding_total_usd')
plt.show()
