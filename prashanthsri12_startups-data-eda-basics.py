# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = 'unicode_escape')

data.head()
data.columns = data.columns.str.strip()
data.columns
data.info()
data.describe()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(data.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
data.dropna(how = 'all',inplace = True)

data.drop(['permalink','homepage_url'],axis=1,inplace = True)

data.shape
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(data.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
data.funding_total_usd = data.funding_total_usd.replace('-','0')
sns.heatmap(data.corr())

plt.show()
def year_segment(x):

    if x is None:

        return x

    elif (x) < 1910:

        return 'before 1910'

    elif (x) < 1930:

        return '1910 to 1930'

    elif (x) < 1950:

        return '1930 to 1950'

    elif (x) < 1970:

        return '1950 to 1970'

    elif (x) < 1990:

        return '1970 to 1990'

    elif (x) < 2010:

        return '1990 to 2010'

    elif (x) >= 2010:

        return 'after 2010'

    



data['year_segment'] = data['founded_year'].apply(lambda x: year_segment(x))   
sns.countplot(data = data, x = data.year_segment)

plt.xticks(rotation = 70)

plt.show()
founded_year_df = data[(data.founded_year > 1990)][['founded_year']].astype(int)

fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.countplot(data = founded_year_df, x = founded_year_df.founded_year)

plt.show()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.countplot(data.market, order = data.market.value_counts()[:20].index)

plt.xticks(rotation = 50)

plt.show()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.countplot(data.country_code, order = data.country_code.value_counts()[:10].index)

plt.xticks(rotation = 50)

plt.show()
fig,axes = plt.subplots(1,1,figsize=(10,5))

sns.countplot(data[data['founded_year']>= 1990]['year_segment'], hue = data.status)

plt.show()
def clear_str(x):

    x = x.replace(',','').strip()

    x = x.replace('-','0')

    return int(x)
data.funding_total_usd = data.funding_total_usd.apply(lambda x: clear_str(x))
market_fund = data.groupby('market').sum()['funding_total_usd'].sort_values(ascending=False)[:10]
market_fund
fig,axes = plt.subplots(1,1,figsize=(20,8))

sns.barplot(x = market_fund.index, y= market_fund.values)

plt.xticks(rotation = 50)

plt.show()
market_closed = data[data['status']=='closed'].groupby('market').count()['name'].sort_values(ascending=False)[:10]
market_closed
fig,axes = plt.subplots(1,1,figsize=(20,8))

sns.barplot(x = market_closed.index, y= market_closed.values)

plt.xticks(rotation = 50)

plt.show()
market_closed_country = data[data['status']=='closed'].groupby('country_code').count()['name'].sort_values(ascending=False)[:10]
market_closed_country
fig,axes = plt.subplots(1,1,figsize=(20,8))

sns.barplot(x = market_closed_country.index, y= market_closed_country.values)

plt.xticks(rotation = 50)

plt.show()