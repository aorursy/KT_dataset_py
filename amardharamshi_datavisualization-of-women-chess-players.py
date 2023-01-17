import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns
dataset = pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv')

dataset
dataset.head()
dataset.shape
dataset.drop(['Fide id'],axis=1)
dataset.rename(columns={'Federation':'Country'},inplace=True)

dataset.head()
dataset.isnull().sum()
dataset['Year_of_birth'].mean()
fig = px.histogram(dataset, "Year_of_birth", nbins=25, width=700)

fig.show()
fig = px.box(dataset, y="Year_of_birth")

fig.show()
dataset['Year_of_birth']=dataset['Year_of_birth'].fillna('1988')
dataset.isnull().sum()
dataset
ds=dataset['Title'].value_counts().reset_index()

ds.columns = ['Title', 'count']

fig = px.bar(ds, x='Title', y="count", orientation='v', title='Title bar chart', width=500)

fig.show()
dataset.dropna(subset=['Title'],inplace=True)
dataset
dataset.isnull().sum()
ds = dataset['Country'].value_counts().reset_index().head(10)

ds.columns = ['Country', 'count']

fig = px.bar(ds, x='Country', y="count", orientation='v', title='Country Having Max Player', width=500,labels={"count":"no. players"})

fig.show()
#top 5 countries#

dataset['Country'].value_counts().head()
#best player of india

dataset[dataset['Country']=='IND'].head(1)
#best player of russia

dataset[dataset['Country']=='RUS'].head(1)
#best player of Germany

dataset[dataset['Country']=='GER'].head(1)
#best player of poland

dataset[dataset['Country']=='POL'].head(1)
#best player of Ukrine

dataset[dataset['Country']=='UKR'].head(1)
#world's best 5 women chess player

dataset.head()
#best 5 players with respect to Standard_Rating , Rapid_rating , Blitz_rating

filtered_df = dataset[dataset['Rapid_rating'].notnull()]

filtered_df.head()
#max rating counties

df1=dataset[dataset['Standard_Rating']>2500]

df2=pd.DataFrame(df1.groupby('Country')['Standard_Rating'].count().sort_values(ascending=False))

df2.columns=['MaxRating']

indx=df2.index





fig = px.bar(x=indx, y=df2['MaxRating'], orientation='v', title='max rating', width=500,labels={"x":"Countries","y":"Most Rated Player"})

fig.show()