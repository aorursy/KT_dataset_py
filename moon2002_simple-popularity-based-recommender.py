import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ratings = pd.read_csv('../input/restaurant-and-consumer-data/rating_final.csv')

cuisine = pd.read_csv('../input/restaurant-and-consumer-data/chefmozcuisine.csv')

geoplaces = pd.read_csv('../input/restaurant-and-consumer-data/geoplaces2.csv',encoding='latin-1')
ratings.head()
rating = pd.DataFrame(ratings.groupby('placeID')['rating'].mean())

rating.head()
rating['ratings_count'] = pd.DataFrame(ratings.groupby('placeID')['rating'].count())

rating
rating.sort_values('ratings_count', ascending=False).head()
top_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])

summary = pd.merge(top_rated_places, cuisine, on='placeID')

summary
cuisine['Rcuisine'].describe()
geoplaces.head()
places = geoplaces[['placeID', 'name']]

places.head()
summary2 = pd.merge(summary, places, on='placeID')

summary2
summary3 = pd.merge(summary2, rating, on='placeID')

summary3
places_crosstab = pd.pivot_table(data=ratings, values='rating', index='userID', columns='placeID')

places_crosstab.head()
Tortas_ratings = places_crosstab[135085]

Tortas_ratings[Tortas_ratings>=0]
Tortas_similarity = places_crosstab.corrwith(Tortas_ratings)

corr_Tortas = pd.DataFrame(Tortas_similarity, columns=['PearsonR'])

corr_Tortas.dropna(inplace=True)

corr_Tortas.head()
Tortas_corr_summary = corr_Tortas.join(rating['ratings_count'])
Tortas_corr_summary[Tortas_corr_summary['ratings_count']>=10].sort_values('PearsonR', ascending=False).head(10)
#places need more than one reviewer in common -  the first 3 rows are not significant as they result from only one common user giving them both the same score
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])

summary_data_table = pd.merge(places_corr_Tortas, cuisine, on='placeID')

summary_data_table
#recommend the other Fast food place

places[places['placeID']==135046]