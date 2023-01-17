# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import pairwise_distances
#https://medium.com/sfu-cspmp/recommendation-systems-user-based-collaborative-filtering-using-n-nearest-neighbors-bf7361dc24e0

import pandas as pd



movies = pd.read_csv("/kaggle/input/movie-lens-small-latest-dataset/movies.csv",encoding="Latin1")

Ratings = pd.read_csv("/kaggle/input/movie-lens-small-latest-dataset/ratings.csv")

Tags = pd.read_csv("/kaggle/input/movie-lens-small-latest-dataset/tags.csv",encoding="Latin1")



Mean = Ratings.groupby(by="userId",as_index=False)['rating'].mean()

Rating_avg = pd.merge(Ratings,Mean,on='userId')

Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']

Rating_avg.head()

movies.head()

Ratings.head()

Tags.head()

check = pd.pivot_table(Rating_avg,values='rating_x',index='userId',columns='movieId')

check.head()
final = pd.pivot_table(Rating_avg,values='adg_rating',index='userId',columns='movieId')

final.head()
# Replacing NaN by Movie Average

final_movie = final.fillna(final.mean(axis=0))



# Replacing NaN by user Average

final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
final_movie.head()

final_user.head()

# user similarity on replacing NAN by user avg

b = cosine_similarity(final_user)

np.fill_diagonal(b, 0 )

similarity_with_user = pd.DataFrame(b,index=final_user.index)

similarity_with_user.columns=final_user.index

similarity_with_user.head()
# user similarity on replacing NAN by item(movie) avg

cosine = cosine_similarity(final_movie)

np.fill_diagonal(cosine, 0 )

similarity_with_movie = pd.DataFrame(cosine,index=final_movie.index)

similarity_with_movie.columns=final_user.index

similarity_with_movie.head()


def find_n_neighbours(df,n):

    order = np.argsort(df.values, axis=1)[:, :n]

    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)

           .iloc[:n].index, 

          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)

    return df
# top 30 neighbours for each user

sim_user_30_u = find_n_neighbours(similarity_with_user,30)

sim_user_30_u.head()


# top 30 neighbours for each user

sim_user_30_m = find_n_neighbours(similarity_with_movie,30)

sim_user_30_m.head()
def get_user_similar_movies( user1, user2 ):

    common_movies = Rating_avg[Rating_avg.userId == user1].merge(

    Rating_avg[Rating_avg.userId == user2],

    on = "movieId",

    how = "inner" )

    return common_movies.merge( movies, on = 'movieId' )


a = get_user_similar_movies(370,86309)

a = a.loc[ : , ['rating_x_x','rating_x_y','title']]

a.head()
def User_item_score(user,item):

    a = sim_user_30_m[sim_user_30_m.index==user].values

    b = a.squeeze().tolist()

    c = final_movie.loc[:,item]

    d = c[c.index.isin(b)]

    f = d[d.notnull()]

    avg_user = Mean.loc[Mean['userId'] == user,'rating'].values[0]

    index = f.index.values.squeeze().tolist()

    corr = similarity_with_movie.loc[user,index]

    fin = pd.concat([f, corr], axis=1)

    fin.columns = ['adg_score','correlation']

    fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)

    nume = fin['score'].sum()

    deno = fin['correlation'].sum()

    final_score = avg_user + (nume/deno)

    return final_score


score = User_item_score(320,7371)

print("score (u,i) is",score)


Rating_avg = Rating_avg.astype({"movieId": str})

Movie_user = Rating_avg.groupby(by = 'userId')['movieId'].apply(lambda x:','.join(x))
def User_item_score1(user):

    Movie_seen_by_user = check.columns[check[check.index==user].notna().any()].tolist()

    a = sim_user_30_m[sim_user_30_m.index==user].values

    b = a.squeeze().tolist()

    d = Movie_user[Movie_user.index.isin(b)]

    l = ','.join(d.values)

    Movie_seen_by_similar_users = l.split(',')

    Movies_under_consideration = list(set(Movie_seen_by_similar_users)-set(list(map(str, Movie_seen_by_user))))

    Movies_under_consideration = list(map(int, Movies_under_consideration))

    score = []

    for item in Movies_under_consideration:

        c = final_movie.loc[:,item]

        d = c[c.index.isin(b)]

        f = d[d.notnull()]

        avg_user = Mean.loc[Mean['userId'] == user,'rating'].values[0]

        index = f.index.values.squeeze().tolist()

        corr = similarity_with_movie.loc[user,index]

        fin = pd.concat([f, corr], axis=1)

        fin.columns = ['adg_score','correlation']

        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)

        nume = fin['score'].sum()

        deno = fin['correlation'].sum()

        final_score = avg_user + (nume/deno)

        score.append(final_score)

    data = pd.DataFrame({'movieId':Movies_under_consideration,'score':score})

    top_5_recommendation = data.sort_values(by='score',ascending=False).head(5)

    Movie_Name = top_5_recommendation.merge(movies, how='inner', on='movieId')

    Movie_Names = Movie_Name.title.values.tolist()

    return Movie_Names
#user = int(input("Enter the user id to whom you want to recommend : "))

user = int(370)

predicted_movies = User_item_score1(user)

print(" ")

print("The Recommendations for User Id : 370")

print("   ")

for i in predicted_movies:

    print(i)