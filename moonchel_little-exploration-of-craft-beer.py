%matplotlib inline

import pandas as pd

beers = pd.read_csv('../input/beers.csv')

breweries = pd.read_csv('../input/breweries.csv')

beers.head()
beers.drop('Unnamed: 0', axis=1, inplace=True)
breweries.rename(columns={'Unnamed: 0' : 'brewery_id'}, inplace=True)

beers.drop('id', axis=1, inplace=True)
beers.head()
breweries.head()
beers = beers.merge(breweries, on='brewery_id')
beers.head()
beers['abv'] *= 100

beers['ibu'].fillna(0.0, inplace=True)

beers['abv'].fillna(0.0, inplace=True)
beers.head()
beers.rename(columns={'name_x' : 'beer', 'name_y' : 'brewery'}, inplace=True)

beers.drop('brewery_id', axis=1, inplace=True)
beers.head()
beers.state.value_counts().head()
beers[['abv', 'state']].sort_values(by='abv', ascending=False).head()
beers.groupby('state')['abv'].mean().sort_values(ascending=False).plot(kind='bar', ylim=(5,7), colormap='summer')
beers[beers['state'].str.contains('CO')].sort_values(by='abv', ascending=False).head()
beers.head()
beers.groupby('city')['brewery'].nunique().sort_values(ascending=False).head().plot(kind='bar', title='Yep, you can.', colormap='summer')
beers.groupby('city')['brewery'].nunique().sort_values(ascending=False).head()
beers.groupby('city')['beer'].nunique().sort_values(ascending=False).head()
beers[beers['city'].str.contains('Grand Rapids')].drop(['ounces', 'state', 'city'], axis=1)