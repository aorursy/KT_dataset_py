import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/winemag-data_first150k.csv', index_col=0)
data.shape
data = data.drop_duplicates('description')
data.shape
data.head()
data.isnull().sum()
data.dtypes
data.dropna(subset=['price','country'],inplace=True)
data.shape
data[['price','points']].corr()
max_price = data.loc[[data['price'].idxmax()]]
max_price
max_score =data[data['points']==100]
print("Number of wines with perfect scores: " + str(len(max_score)))
print('\n')
print('List of wines with perfect scores, price low to high :')
max_score[['country', 'price', 'province', 'variety', 'winery']].sort_values(by=['price'])
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
data['points'].hist(bins=30, alpha=.65)
plt.xlabel('Points')
plt.figure(figsize=(10,6))
data['price'].hist(bins=50, alpha=.65)
plt.xlabel('Price $')
import math
data['rounded_price']= data['price'].apply(lambda x: math.ceil(x/10.0)*10)
plt.figure(figsize=(20,10))
data['rounded_price'].value_counts().sort_index().plot.bar()
plt.xlabel('Rounded Price $')
plt.ylabel('count')
print('Countries in dataset: ' + '\n' + str(sorted(list(data['country'].unique()))))
print('\n')
print('Number of unique Countries in dataset: ' + str(data['country'].nunique()))
df = pd.DataFrame(data.groupby('country')['points','price'].mean())
df1 = pd.DataFrame(data.groupby('country')['description'].count())
country= df.join(df1)
country.columns= ['Avg. Points', 'Avg. Price', 'Count of Wine']
country
country= country[country['Count of Wine']>=30]
country.shape
country.sort_values(by=['Avg. Points'],inplace=True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

trace1 = go.Bar(
    x= country.index,
    y= country['Avg. Points'],
    name='Avg. Points'
)

trace2= go.Bar(
    x= country.index,
    y= country['Avg. Price'],
    name='Avg. Price'
)
data=[trace1, trace2]
layout=go.Layout(
    barmode='stack')

fig=go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
