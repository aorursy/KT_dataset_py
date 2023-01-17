import pandas as pd
import numpy as np
frame = pd.read_csv('../input/rating_final.csv')
cuisine = pd.read_csv('../input/chefmozcuisine.csv')

#Ratings dataframe
frame.head()
#Cuisine type
cuisine.head()
#Popularity based recommendation
#Grouping based on rating counts

rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())
rating_count.sort_values('rating', ascending=False).head()
#Most rated places' cuisine type
most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])

summary = pd.merge(most_rated_places, cuisine, on='placeID')
summary
cuisine['Rcuisine'].describe()
#Correlation based recommendation
#geo data
geodata = pd.read_csv('../input/geoplaces2.csv', encoding = 'latin-1')
places =  geodata[['placeID', 'name']]
places.head()
#Grouping and Ranking Data
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())

rating.head()
rating.describe()
rating.sort_values('rating_count', ascending=False).head()
#Getting the name of the most popular place
places[places['placeID']==135085]
cuisine[cuisine['placeID']==135085]
#Data prep
#Combining dataframes
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
places_crosstab.head()
#Evaluating Similarity Based on Correlation
Tortas_ratings = places_crosstab[135085]
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)

corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
corr_Tortas.head()
#Getting places that have at least 10 reviews and are correlated with the most popular resturant
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10)
#Creating a list of top correlated places 
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])
#Creating a summary table
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID')
summary
