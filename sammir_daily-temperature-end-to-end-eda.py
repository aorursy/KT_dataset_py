import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.display import HTML



import warnings
%matplotlib inline

plt.style.use('ggplot')

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
df.shape
df.shape[0]
df.shape[1]
df.columns
df.head()
df.tail()
df.dtypes
df['Day_STR']=df['Day'].astype(str)

df['Month_STR']=df['Month'].astype(str)

df['Year_STR']=df['Year'].astype(str)
df.dtypes
df.info()
df.isna().sum()
fig, ax = plt.subplots(figsize=(16,8))

sns.boxplot(x='Region',y='AvgTemperature',data=df,ax=ax)
# Check the number of records

df[df['AvgTemperature']==-99.0].count()
# Remove these records

df=df.drop(df[df['AvgTemperature']==-99.0].index)
df.head()
df['Date']=df['Day'].astype(str)+'/'+df['Month'].astype(str)+'/'+df['Year'].astype(str)
df.head()
df['Date']=pd.to_datetime(df['Date'])
df.dtypes
df.head()
df.shape
df['Country'].nunique()
df.groupby(['Region'])['Country'].nunique()
df.head()
df.groupby(['Region'])['AvgTemperature'].mean()
pivoted_df=pd.pivot_table(df[['Region','AvgTemperature','Year']], 

                          values='AvgTemperature', index=['Region'],

                          columns=['Year'], aggfunc=np.mean)
pivoted_df.head()
pivoted_df.to_csv('region_temperature_racing_bar_chart.csv')
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2738038" data-url="https://flo.uri.sh/visualisation/2738038/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
df.groupby(['Region'])['AvgTemperature'].mean().plot(kind='bar',figsize=(17,7))
df.groupby(['Region','Country'])['AvgTemperature'].max().sort_values(ascending=False).head(10)
df.groupby(['Country'])['AvgTemperature'].min().sort_values(ascending=False).head(10)
pivoted_Country_Year_df=pd.pivot_table(df[['Country','AvgTemperature','Year']], 

                          values='AvgTemperature', index=['Country'],

                          columns=['Year'], aggfunc=np.mean)
pivoted_Country_Year_df.head()
pivoted_Country_Year_df.to_csv('country_temperature_racing_bar_chart.csv')
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2738445" 

data-url="https://flo.uri.sh/visualisation/2738445/embed">

<script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].min()
df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].max()
df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].mean()
df.groupby('City')['AvgTemperature'].max().sort_values(ascending=False).head(10)
df.groupby('City')['AvgTemperature'].max().sort_values(ascending=False).head(30).plot(kind='bar',figsize=(17,7))
df[df['Region']=='Asia'].groupby(['Country','Year','Month','Day'])['AvgTemperature'].max().sort_values(ascending=False).head()
df.groupby(df['Date'].dt.to_period('M'))['AvgTemperature'].mean().plot(kind='line',figsize=(17,7))
df['AvgTemperature'].max()
df['AvgTemperature'].min()
df['AvgTemperature'].mean()
df['AvgTemperature'].std()
df['AvgTemperature'].var()
df['AvgTemperature'].skew()
df['AvgTemperature'].kurt()
sns.distplot(df['AvgTemperature'])
df.describe(include='all')
sns.relplot(x='Year',y='AvgTemperature',data=df,kind='line',hue='Region',height=16, aspect=1)
sns.pairplot(data=df[['AvgTemperature','Region','Month']],hue='Region',height=7, aspect=1)