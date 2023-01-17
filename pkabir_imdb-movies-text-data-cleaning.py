import pandas as pd
import numpy as np

movies = pd.read_csv('../input/imdb_1000.csv', header=0)
movies.head()
movies.actors_list.replace(['\[', 'u\'','\'\]'],'', regex=True, inplace=True)
movies.actors_list.head()
movies.actors_list.replace('\',', ',', regex=True, inplace=True)
movies.actors_list.unique()
actors=pd.DataFrame(movies.actors_list.str.split(',').tolist(), columns = ['actor_1','actor_2','actor_3'])
#actors.head()
movies=pd.concat([movies, actors], axis=1)
movies.head()
movies.drop('actors_list', axis=1, inplace=True)
movies.sample(5)
movies.to_csv("clean_text.csv")