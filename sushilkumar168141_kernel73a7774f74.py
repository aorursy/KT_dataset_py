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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head(10)
df.info()
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
df.head(10)
df.isnull().sum()
df['Platform'].nunique()
df['Platform'].unique()
df['Genre'].nunique()
df['Genre'].unique()
df['Publisher'].nunique()
Sales_by_platform = df.groupby(by='Platform').sum()
Sales_by_platform.drop(['Rank','Year'], axis=1, inplace=True)
Sales_by_platform.reset_index(inplace=True)
Sales_by_platform
Sales_by_platform.describe()
# Platform Wise sales in various countries

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20,20))

sns.barplot(x='Platform', y='NA_Sales', data=Sales_by_platform.sort_values(by='NA_Sales', ascending=False), ax=ax[0])

sns.barplot(x='Platform', y = 'EU_Sales', data=Sales_by_platform.sort_values(by='EU_Sales', ascending=False), ax=ax[1])

sns.barplot(x='Platform',y = 'JP_Sales', data=Sales_by_platform.sort_values(by='JP_Sales', ascending=False), ax=ax[2] )

sns.barplot(x='Platform', y ='Other_Sales', data=Sales_by_platform.sort_values(by='Other_Sales', ascending=False), ax=ax[3])

sns.barplot(x='Platform', y = 'Global_Sales', data=Sales_by_platform.sort_values(by='Global_Sales', ascending=False), ax=ax[4])
year_wise_sales = pd.DataFrame(df.groupby(by='Year').sum())
year_wise_sales.drop(['Rank'], axis=1, inplace=True)
year_wise_sales.reset_index(inplace=True)
year_wise_sales.head()
# visualisation for year_wise different sales

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20,30))

sns.barplot(x='Year', y='NA_Sales', data=year_wise_sales, ax=ax[0])

sns.barplot(x='Year', y='EU_Sales', data=year_wise_sales, ax=ax[1])

sns.barplot(x='Year', y='JP_Sales', data=year_wise_sales, ax=ax[2])

sns.barplot(x='Year', y='Other_Sales', data=year_wise_sales, ax=ax[3])

sns.barplot(x='Year', y='Global_Sales', data=year_wise_sales, ax=ax[4])

# Genre wise sales 

genre_wise_sales = df.groupby(by='Genre').sum()
genre_wise_sales.drop(['Rank','Year'], axis=1, inplace=True)
genre_wise_sales.reset_index(inplace=True)
genre_wise_sales
# Visulaisation for genre wise different sales

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20,20))

sns.barplot(x='Genre', y='NA_Sales', data=genre_wise_sales.sort_values(by='NA_Sales', ascending=False), ax=ax[0])

sns.barplot(x='Genre', y='EU_Sales', data=genre_wise_sales.sort_values(by='EU_Sales', ascending=False), ax=ax[1])

sns.barplot(x='Genre', y='JP_Sales', data=genre_wise_sales.sort_values(by='JP_Sales', ascending=False), ax=ax[2])

sns.barplot(x='Genre', y='Other_Sales', data=genre_wise_sales.sort_values(by='Other_Sales', ascending=False), ax=ax[3])

sns.barplot(x='Genre', y='Global_Sales', data=genre_wise_sales.sort_values(by='Global_Sales', ascending=False), ax=ax[4])
# Publisher wise total sales in different country

publisher_wise_sales=df.groupby('Publisher').sum()
publisher_wise_sales.drop(['Rank','Year'], axis=1, inplace=True)
publisher_wise_sales.reset_index(inplace=True)
publisher_wise_sales
df.head()
# Sales for publisher in different year

df.groupby(['Year','Publisher']).sum().drop('Rank', axis=1)
# Salses for  Platform in different year

df.groupby(['Year','Platform']).sum().drop('Rank', axis=1)
# Sales for Genre in different Years

df.groupby(['Year','Genre']).sum().drop('Rank', axis=1)
df.sort_values('Global_Sales', ascending=False)
profiling_report=pandas_profiling.ProfileReport(df)
profiling_report
profiling_report.to_file('Video_game_sales_EDA.html')