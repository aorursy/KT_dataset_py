import pandas as pd 

import numpy as np 

import requests 

import json

import matplotlib.pyplot as plt

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("myapikey")



responses = [] 

movie_ids = ['tt0096874','tt0482571','tt0497465','tt0325980','tt0383574','tt0449088','tt0407887',

'tt0431308','tt2872732','tt0120855','tt0892769','tt4633694',

'tt1646971','tt0097165','tt0362227','tt1979320','tt2322441',

'tt1411238','tt1727824','tt6105098','tt0808151','tt0172495',

'tt0974661','tt0443706','tt1403981','tt3890160','tt0092890',

'tt3521164','tt0332379','tt3606756','tt7286456','tt1059955',

'tt1424432','tt9243946','tt7455754','tt0114369','tt4729430','tt1950186','tt7139936']





for movie in movie_ids:

   rr = requests.get(secret_value_0.format(movie))

   data=json.loads(rr.text)

   responses.append(data)
movies_df = pd.DataFrame(responses)

movies_df
movies_df.dtypes
#Dropping unnecessary columns 

movies_df = movies_df.drop(columns = ['Year','Website', 'Response', 'Ratings', 'Type' ])



#Removing 'min' from 'Runtime' column and '$' from 'BoxOffice' in order to convert them into integers and floats respectively.

movies_df['Runtime'] = movies_df['Runtime'].str.replace('min','')

movies_df['BoxOffice'] = movies_df['BoxOffice'].str.replace('$','')



#Removing ',' from 'BoxOffice' and 'imdbVotes' in order to be able to convert them into numeric values.

movies_df['BoxOffice'] = movies_df['BoxOffice'].str.replace(',','')

movies_df['imdbVotes'] = movies_df['imdbVotes'].str.replace(',','')

movies_df = movies_df.replace('N/A', np.nan)



#Renaming the two columns to make the values interpretable. 

movies_df.rename(columns={'Runtime': 'Runtime (min)', 'BoxOffice': 'BoxOffice($)'}, inplace=True)



#Converting the columns that contain numeric values to integers or floats. First we have to fill all the NaN values 

#with 0 because the 'pd.to_numeric' function will convert columns with NaN values to floats. 



movies_df[['imdbRating', 'imdbVotes', 'BoxOffice($)', 'Runtime (min)', 'Metascore']] = movies_df[['imdbRating', 

                                                                                                  'imdbVotes', 

                                                                                                  'BoxOffice($)', 

                                                                                                  'Runtime (min)', 

                                                                                                  'Metascore']].fillna(0).apply(pd.to_numeric)



movies_df['Released'] = pd.to_datetime(movies_df['Released'])

movies_df['DVD'] = pd.to_datetime(movies_df['DVD'])

movies_df['Rated'] = movies_df['Rated'].astype('category')
movies_df.dtypes
#Turning strings into lists. 

movies_df['Genres_new'] = movies_df['Genre'].str.split(', ')   
#Counting the number of times each genre appears in the list 

genres_count = pd.Series(sum([item for item in movies_df.Genres_new], [])).value_counts()
genres_count.plot(kind='barh', figsize = (10,10), xticks = range(0 ,21, 2))

plt.xlabel('Count')

plt.ylabel('Genres')

plt.title('Most preferred genres')

plt.gca().invert_yaxis()

plt.show()
movies_df['Actors_new'] = movies_df['Actors'].str.split(', ')   
actors_count = pd.Series(sum([item for item in movies_df.Actors_new], [])).value_counts()
actors_count[0:20].plot(kind='barh', figsize=(10,10), xticks= range(0,4,1))

plt.xlabel('Count')

plt.ylabel('Genres')

plt.title('Actors we watched the most')

plt.gca().invert_yaxis()

plt.show()
movies_df['Runtime (min)'].mean()
movies_df['imdbRating'].mean()
movies_df[movies_df['imdbRating'] == movies_df['imdbRating'].max()]
movies_df[movies_df['imdbRating'] == movies_df['imdbRating'].min()]
plt.scatter('Runtime (min)', 'imdbRating', data=movies_df)

plt.xlabel('Runtime (min)')

plt.ylabel('IMDB Rating')

plt.show()
plt.scatter('Metascore', 'imdbRating', data=movies_df)

plt.xlabel('Metascore')

plt.ylabel('IMDB Rating')

plt.show()
plt.scatter('imdbVotes', 'imdbRating', data=movies_df)

plt.xlabel('Number of votes on IMDB')

plt.ylabel('IMDB Rating')

plt.show()
plt.scatter('Released', 'imdbRating', data=movies_df)

plt.xlabel('Year of release')

plt.ylabel('IMDB Rating')

plt.show()
movies_df['Production'].value_counts()