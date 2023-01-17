import numpy as np

import pandas as pd

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns
zomato1 = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")

country_code = pd.read_excel('../input/Country-Code.xlsx')

zomato = pd.merge(zomato1,country_code, on='Country Code') #Объединим zomato1 и country_code в zomato, чтобы иметь код страны и её название в одном наборе данных.

zomato.head().T
zomato['Country'].unique()
zomato['Country'].value_counts().head()
zomato['City'].unique()
zomato['City'].value_counts().head()
zomato['Country'].value_counts().head().plot(kind='bar', title='Number of zomato restaurants in top-5 countries')
zomato['City'].value_counts().head().plot(kind='bar',title='Number of zomato restaurants in top-5 cities')
def func(s):

    return s.split(', ')



cuisines_list1 = zomato['Cuisines'].unique()

cuisines_list2 = []

for i in range (cuisines_list1.size):

    cuisines_list2 += func(str(cuisines_list1[i]))

cuisines_list = list(set(cuisines_list2))

cuisines_list 
zomato.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
zomato.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(50).plot(kind='bar',figsize=(20,7),title='Aggregate rating top-50 cuisines')
from scipy.stats import pearsonr, spearmanr, kendalltau

r = spearmanr(zomato['Price range'], zomato['Aggregate rating'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
zomato.groupby('Price range')['Aggregate rating'].mean().sort_values(ascending=False)
zomato.groupby('Price range')['Aggregate rating'].mean().sort_values(ascending=False).plot(kind='bar',title='Relation between Price range and Aggregate rating')
new_values = {'Yes':1, 'No':0}

zomato['Dummy Has Table booking'] = zomato['Has Table booking'].map(new_values)



from scipy.stats import pointbiserialr

pointbiserialr(zomato['Aggregate rating'], zomato['Dummy Has Table booking'])
zomato.groupby('Has Table booking')['Aggregate rating'].mean().sort_values(ascending=False)
zomato.groupby('Has Table booking')['Aggregate rating'].mean().sort_values(ascending=False).plot(kind='bar',title='Relation between Booking and Aggregate rating')
colorscale = [

[0, 'rgb(255, 255, 255)'], 

[0.1, 'rgb(255, 255, 255)'],

[0.2, 'rgb(212, 28, 28)'], 

[0.3, 'rgb(212, 28, 28)'],

[0.4, 'rgb(255, 186, 59)'], 

[0.5, 'rgb(255, 186, 59)'],

[0.6, 'rgb(248, 255, 59)'], 

[0.7, 'rgb(248, 255, 59)'], 

[0.8, 'rgb(0, 227, 30)'], 

[0.9, 'rgb(0, 227, 30)'],

[1, 'rgb(0, 117, 16)']

]



fig1 = [dict(

    type='scattergeo',

    lon = zomato['Longitude'],

    lat = zomato['Latitude'],

    text = zomato['Restaurant Name'],

    mode = 'markers',

    marker = dict(

    color = zomato['Rating color'],

    cmin = zomato['Aggregate rating'].min(),

    cmax = zomato['Aggregate rating'].max(),

    colorscale=colorscale,

    colorbar = dict(

                title = 'Rating'

            )

        )

    )]



fig_layout = dict(

    title = 'Rating of zomato restaurants on the world map'

)

fig = go.Figure(data=fig1, layout=fig_layout)

iplot(fig)
max_votes = zomato['Votes'].sort_values(ascending=False).head(20)

zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']]
max_votes = zomato['Votes'].sort_values(ascending=False).head(20)

zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']].plot(kind='bar', x = 'Restaurant Name', y = 'Votes', figsize=(20,10), color = 'darkblue')
zomato_india = zomato.loc[zomato['Country']=='India']



from scipy.stats import pearsonr, spearmanr, kendalltau

r = spearmanr(zomato_india['Average Cost for two'], zomato_india['Aggregate rating'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
zomato_india.plot.scatter(x='Average Cost for two',y='Aggregate rating',figsize=(15,7), title="Cost vs Agg Rating")
zomato['Cuisines'].value_counts()
zomato['Cuisines'].value_counts().head(20).plot(kind='bar', figsize = (20,7))
zomato['Restaurant Name'].value_counts()
zomato['Restaurant Name'].value_counts().head(20).plot(kind='pie',figsize=(15,15), title="Top 15 Restaurants with maximum outlets", autopct='%1.2f%%')
numeric = ['Price range', 'Aggregate rating', 'Votes', 'Longitude', 'Latitude']

sns.pairplot(zomato[numeric])
sns.boxplot(zomato['Price range']);
sns.boxplot(zomato['Aggregate rating']);
sns.boxplot(zomato['Votes']);
sns.boxplot(zomato['Longitude']);
sns.boxplot(zomato['Latitude']);
def outliers_indices(feature):

    mid = zomato[feature].mean()

    sigma = zomato[feature].std()

    return zomato[(zomato[feature] < mid - 3*sigma) | (zomato[feature] > mid + 3*sigma)].index



wrong_price_range = outliers_indices('Price range')

wrong_aggregate_rating = outliers_indices('Aggregate rating')

wrong_votes = outliers_indices('Votes')

wrong_longitude = outliers_indices('Longitude')

wrong_latitude = outliers_indices('Latitude')



out = set(wrong_price_range) | set(wrong_aggregate_rating) | set(wrong_votes) | set(wrong_longitude) | set(wrong_latitude)

print(len(out))
zomato.drop(out, inplace=True)
zomato[numeric].corr(method='spearman')
sns.heatmap(zomato[numeric].corr(method='spearman'));