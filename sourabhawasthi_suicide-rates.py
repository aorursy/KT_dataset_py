import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
%matplotlib inline
suicide_df = pd.read_csv('../input/master.csv')
suicide_df.head()
suicide_df.info()
suicide_df.describe()
suicide_df.isnull().any()
sns.catplot('country','population',hue='age',data=suicide_df)
suicide_df = suicide_df.drop(['HDI for year','country-year','gdp_per_capita ($)'],axis=1)
min_year = min(suicide_df.year)
max_year = max(suicide_df.year)
print('Max year :',max_year)
print('Min year :',min_year)
#df = suicide_df.groupby(['country']).sum()
df = suicide_df[['country', 'suicides_no']]
df.head()
df1 = df.groupby('country').sum()
df1 = df1.sort_values(by='suicides_no', ascending=False).reset_index()
df1 = df1.loc[df1['suicides_no'] > 1000]
df1.head()
plt.figure(figsize=(15,20))
sns.barplot(x='suicides_no',y='country',data=df1)
suicide_df.groupby('sex')['suicides_no'].sum().plot(kind='bar', cmap='RdBu')
suicide_df.groupby('age')['suicides_no'].sum().plot(kind='bar', cmap='rainbow')
suicide_df.groupby('year')['suicides_no'].sum().plot(kind='bar',figsize=(15,10), cmap='summer')
pop = suicide_df[['country','population','suicides_no']]
pop.head()
df1.sort_values(by='suicides_no',inplace= True,ascending= False)
plt.figure(figsize=(10,6))
sns.countplot(x='generation', hue='sex',data= suicide_df)
suicide_df.plot(x='generation',y='suicides_no',linestyle='',marker='o')
