import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
print(data.columns)
data.index
data.isnull().sum()
data['director'].value_counts()
data.drop(columns=['director'],inplace=True)
data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
data.drop('cast',inplace=True,axis=1)
data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
countries=data['country']
countries
unique_country=data['country'].unique()
unique_country
occurences=np.count_nonzero(countries,axis=0)
occurences
(data.country=='United States').value_counts()
(data.country=='United Kingdom').value_counts()
(data.country=='India').value_counts()
data['country'].value_counts()
data.dtypes
data['type'].unique()
pd.get_dummies(data['type'])
data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
sns.heatmap(data.isnull())
data[data['type']=='TV Show']
data[data['type']=='TV Show'].value_counts()
sns.countplot(data=data,y='country',hue='type')
data.fillna('bfill',inplace=True)
data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
sns.heatmap(data.isnull())
data.isnull().sum()
sns.pairplot(data)
data.hist(figsize=(20,15))
sns.distplot(data['release_year'])
sns.kdeplot(data['release_year'])
sns.countplot(data=data,x='release_year',hue='type')
sns.barplot(data=data,x='release_year',y='type')
data['type']=data['type'].astype('category')
data.columns
sns.jointplot(data=data,x='release_year',y='show_id',kind='reg')
sns.jointplot(data=data,x='release_year',y='show_id',kind='kde')
data.corr()
sns.heatmap(data.corr())
sns.set_style("whitegrid")
import plotly.express as px
fig = px.bar(data,x='release_year',y='country',title='Country Vs release_year', height=900, orientation='h')
fig.show()
