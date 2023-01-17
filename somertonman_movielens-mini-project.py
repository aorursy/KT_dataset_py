import pandas as pd

import collections

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!ls /kaggle/input/movielens-20m-dataset/

#!cat /kaggle/input/movielens-20m-dataset/movie.csv
movies = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv', sep=',')

movies.head(10)
movies.isnull().any()
ratings = pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv', sep=',', parse_dates=['timestamp'])

ratings.head()
ratings.isnull().any()
ratings['rating'].describe()
#filter_1 = ratings['rating'] > 5

movies_with_ratings = movies.merge(ratings, on='movieId', how='inner')
movies_with_ratings.head(10)
#count most frequent genres

collections.Counter(" ".join(movies["genres"]).split("|")).most_common(10)

collections.Counter(" ".join(movies["genres"]).split("|")).most_common()[-10:]
ratings.groupby('userId', as_index=False).count().sort_values(['movieId'], ascending=False)[:10]
average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()

movies_avg = movies.merge(average_rating, on='movieId', how='inner')

movies_avg.head(10)
movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)

movies.head(10)
is_Drama = movies_avg['genres'].str.contains('Romance',case=False)

movies_avg["rating"][is_Drama].mean()

movies_avg[is_Drama].head(10)
movies_avg["rating"][is_Drama].count()
common_genres = collections.Counter(" ".join(movies["genres"]).split('|')).most_common(10)

print(common_genres)

def counting_genres():

    for i in common_genres:

        gen={}

        is_genre = movies_avg['genres'].str.contains(i[0])

        gen[i[0]]=movies_avg["rating"][is_genre].mean()

    return gen

    



print(counting_genres())
def splitDataFrameList(df,target_column,separator):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    separator = the symbol used to perform the split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    def splitListToRows(row,row_accumulator,target_column,separator):

        split_row = row[target_column].split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df



splitDataFrameList(movies,"genres","|").head(10)
movies_genres_splitted = splitDataFrameList(movies,"genres","|")
movies_genres_splitted_avg = movies_genres_splitted.merge(average_rating, on='movieId', how='inner')
movies_genres_splitted_avg.head(10)
most_popular_genres = movies_genres_splitted_avg[['genres','rating']].groupby('genres', as_index=True).count().sort_values(['rating'], ascending=False)[:10]

most_popular_genres
best_rated= movies_genres_splitted_avg[['genres','rating']].groupby('genres', as_index=False).mean().sort_values(['rating'], ascending=False)[:10]

best_rated
best_rated.plot(x='genres', y='rating', title="Average highest ranked film genres", figsize=(15,8),kind='bar')
cleaned = movies_genres_splitted_avg[pd.to_numeric(movies_genres_splitted_avg['year'], errors='coerce').notnull()]
is_Documentary = cleaned["genres"]=="Documentary"

movies_genres_splitted_avg_documentary = cleaned[is_Documentary]

movies_genres_splitted_avg_documentary[:5]
best_rated_documentaries= movies_genres_splitted_avg_documentary[['year','rating']].groupby('year', as_index=False).mean().sort_values(['year'], ascending=False)

best_rated_documentaries.plot(x='year', y='rating', title="Average ratings for Documentary films per year", figsize=(15,8))
is_Filmnoir = cleaned["genres"]=="Film-Noir"

movies_genres_splitted_avg_filmnoir = cleaned[is_Filmnoir]

movies_genres_splitted_avg_filmnoir[:5]
best_rated_filmnoir= movies_genres_splitted_avg_filmnoir[['year','rating']].groupby('year', as_index=False).mean().sort_values(['year'], ascending=False)

best_rated_filmnoir.plot(x='year', y='rating', title="Average ratings for Film-Noir per year", figsize=(15,8))
is_Thriller = cleaned["genres"]=="Thriller"

movies_genres_splitted_avg_documentary = cleaned[is_Thriller]

movies_genres_splitted_avg_documentary[:5]

best_rated_documentaries= movies_genres_splitted_avg_documentary[['year','rating']].groupby('year', as_index=False).mean().sort_values(['year'], ascending=False)

best_rated_documentaries.plot(x='year', y='rating', title="Average ratings for Thriller films per year", figsize=(15,8))
plt.figure(figsize=(15,8))

plt.plot(best_rated_documentaries['year'], best_rated_documentaries['rating'], label='label here')

plt.plot(best_rated_filmnoir['year'], best_rated_filmnoir['rating'], label='label here')



#plt.plot(<X AXIS VALUES HERE>, <Y AXIS VALUES HERE>, 'line type', label='label here')

#plt.legend(loc='best')

plt.show()