import pandas as pd

import numpy as np



imdb = pd.read_csv('../input/movie_metadata.csv')

imdb.head()
imdb.shape
imdb.isnull().any()
imdb.groupby(['actor_1_name','actor_2_name','actor_3_name','movie_title']).imdb_score.max().sort_values(ascending = False).reset_index().head()
imdb.groupby(['budget','movie_title','imdb_score']).gross.max().sort_values(ascending = False).reset_index().head()
imdb.country.value_counts().head()
imdb.title_year.value_counts().head()
imdb.groupby('director_name').imdb_score.agg(['max','min']).reset_index().head(10)
def score_review(x):

    if x<=3:

        z='Need not watch!'

    elif (x>3 and x<=7):

        z='Good to watch'

    elif x>7 and x<=10:

        z='Must watch'

    else:

        z="."

    return z

 

imdb['score_intrepretation']=imdb.imdb_score.apply(score_review)

imdb.head()
imdb['content_rating'].value_counts().plot(kind='bar')
imdb['imdb_score'] = imdb['imdb_score'].apply(lambda x:int(round(x)))
imdb['imdb_score'].value_counts().plot(kind='bar')
df=pd.DataFrame(imdb[['gross','budget']])

df.plot.line(subplots=True)
imdb.groupby('content_rating').imdb_score.max().plot(kind='bar')
imdb.plot(x='duration',y='imdb_score',kind='scatter')