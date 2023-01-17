import dask.dataframe as dd

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from surprise import Reader

from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise import SlopeOne

from surprise import CoClustering

from surprise import BaselineOnly

from surprise.accuracy import rmse

from surprise import accuracy

from surprise.model_selection import train_test_split

from collections import defaultdict
ratings = dd.read_csv('../input/anime-recommendations-database/rating.csv')

ratings.head(2)
#Para evitar confusiones con los nombres

ratings=ratings.rename(columns = {'rating':'user_rating'})
#importamos la data de anime

anime = dd.read_csv('../input/anime-recommendations-database/anime.csv')

anime.head(2)
#Numero de usuarios unicos 

ratings.user_id.nunique().compute()
#información basica sobre sobre ratings

sns.distplot(ratings.user_rating)

ratings.describe().compute()
users=ratings.groupby('user_id').agg({'user_rating':['mean','count','std']})

users.columns=['media','n_puntuaciones','std_puntuaciones']
sns.distplot(users.media,kde=False)

users.media.describe().compute()
sns.distplot(users.n_puntuaciones)

users.n_puntuaciones.describe().compute()
# Esto significa:

sin_puntuaciones=ratings.groupby(['user_id']).agg({'user_rating':['count','sum']})

print (len(sin_puntuaciones[-sin_puntuaciones['user_rating','sum']==sin_puntuaciones['user_rating','count']]),' usuarios perdidos')
#Nos deshacemos de ambos

ratings=ratings[ratings.user_rating>0]
users=ratings.groupby('user_id').agg({'user_rating':['mean','count','std']})

users.columns=['MPRU','n_puntuaciones','std_puntuaciones']
# %  de perdida de usuarios con solo una puntuación:

100*len(users[users.n_puntuaciones<=1])/len(users)
# % de perdida de usuarios con 2 o menos puntuaciones:

100*len(users[users.n_puntuaciones<=2])/len(users)
# % de perdida de usuarios con 3 o menos puntuaciones:

100*len(users[users.n_puntuaciones<=3])/len(users)
#usuarios con 4 o más puntuaciones

users=users[users.n_puntuaciones>3]
#Juntamos los datasets para tener la informacipon completa

data=dd.merge(ratings,users,on='user_id',how='inner')
df=dd.merge(data[['user_id','anime_id','user_rating']],anime[['anime_id','name']],on='anime_id',how='left',indicator=True)

df.head()
#Revisemos Cuantos no nos cruzaron.

len(data)-len(df[df._merge=='both'])
#Esto significa que hay animes id sin nombre correspondiente.

df.anime_id[df['_merge']=='left_only'].compute().unique()
#dejamos fuera aquellas filas cuyo anime_id no se corresponde entre los dos sets.

df=df[df._merge=='both']
df_triple=df[['user_id','anime_id','user_rating']]
len(df_triple)
reader = Reader(rating_scale=(1.0, 10.0))

data = Dataset.load_from_df(df_triple[['user_id', 'anime_id', 'user_rating']], reader)
#El performance del algoritmo estara dado por una validación cruzada de 5 iteraciones.

#El conjunto de testeo es una valoración por usuario en el dataset.

results = cross_validate(BaselineOnly(), data, measures=['RMSE'], cv=5, verbose=False)

basel = pd.DataFrame.from_dict(results).mean(axis=0)

basel = basel.append(pd.Series([str(BaselineOnly()).split(' ')[0].split('.')[-1]], index=['BaselineOnly()']))
basel
#El performance del algoritmo estara dado por una validación cruzada de 5 iteraciones.

#El conjunto de testeo es una valoración por usuario en el dataset.

results = cross_validate(CoClustering(), data, measures=['RMSE'], cv=5, verbose=False)

coclus = pd.DataFrame.from_dict(results).mean(axis=0)

coclus = coclus.append(pd.Series([str(CoClustering()).split(' ')[0].split('.')[-1]], index=['CoClustering()']))
coclus
#El performance del algoritmo estara dado por una validación cruzada de 5 iteraciones.

#El conjunto de testeo es una valoración por usuario en el dataset.

results = cross_validate(SlopeOne(), data, measures=['RMSE'], cv=5, verbose=False)

slope = pd.DataFrame.from_dict(results).mean(axis=0)

slope = slope.append(pd.Series([str(SlopeOne()).split(' ')[0].split('.')[-1]], index=['SlopeOne()']))
slope
trainset, testset = train_test_split(data, test_size=0.20)

slopeone=SlopeOne()

predict_test = slopeone.fit(trainset).test(testset)

accuracy.rmse(predict_test)
def top(user,predict_test=predict_test,anime=anime):

    anime=anime[['anime_id','name']]

    predichos=dd.from_pandas(pd.DataFrame.from_records(predict_test), npartitions=6)

    predichos=predichos[[0,1,2,3]]

    predichos.columns=['user_id','anime_id','p_real','p_predicha']

    predichos=predichos[predichos.user_id==user]

    predichos.p_predicha=predichos.p_predicha.round(decimals=0)

    predichos=predichos.compute()

    #Ya lo redujimos lo suficiente como para usar pandas con normalidad.

    predichos=pd.DataFrame(predichos.sort_values(by='p_predicha',ascending=False).head(3))

    return(pd.merge(predichos,anime.compute(),on='anime_id',how='inner'))
predichos=dd.from_pandas(pd.DataFrame.from_records(predict_test), npartitions=6)

predichos=predichos[[0,1,2,3]]

predichos.columns=['user_id','anime_id','p_real','p_predicha']
top(33888)
top(28859)
top(5255)
top(65451)