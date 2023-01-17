import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv("../input/movie_metadata.csv")



data_use = data.ix[:,['genres','plot_keywords','movie_title','actor_1_name',

                      'actor_2_name','actor_3_name','director_name','imdb_score']]



data_use['movie_title'] = [i.replace("\xa0","") for i in list(data_use['movie_title'])]
print(data_use.shape)

clean_data = data_use.dropna(axis = 0)

print(clean_data.shape)

clean_data = clean_data.drop_duplicates(['movie_title'])

clean_data = clean_data.reset_index(drop=True)

print(clean_data.shape)
people_list = []

for i in range(clean_data.shape[0]):

    name1 = clean_data.ix[i,'actor_1_name'].replace(" ","_")

    name2 = clean_data.ix[i,'actor_2_name'].replace(" ","_")

    name3 = clean_data.ix[i,'actor_3_name'].replace(" ","_")

    name4 = clean_data.ix[i,'director_name'].replace(" ","_")

    people_list.append("|".join([name1,name2,name3,name4]))

clean_data['people'] = people_list
from sklearn.feature_extraction.text import CountVectorizer



def token(text):

    return(text.split("|"))





cv_kw=CountVectorizer(max_features=100,tokenizer=token )

keywords = cv_kw.fit_transform(clean_data["plot_keywords"])

keywords_list = ["kw_" + i for i in cv_kw.get_feature_names()]



cv_ge=CountVectorizer(tokenizer=token )

genres = cv_ge.fit_transform(clean_data["genres"])

genres_list = ["genres_"+ i for i in cv_ge.get_feature_names()]



cv_pp=CountVectorizer(max_features=100,tokenizer=token )

people = cv_pp.fit_transform(clean_data["people"])

people_list = ["pp_"+ i for i in cv_pp.get_feature_names()]



cluster_data = np.hstack([keywords.todense(),genres.todense(),people.todense()*2])

criterion_list = keywords_list+genres_list+people_list
from sklearn.cluster import KMeans



mod = KMeans(n_clusters=100)

category = mod.fit_predict(cluster_data)

category_dataframe = pd.DataFrame({"category":category},index = clean_data['movie_title'])
clean_data.ix[list(category_dataframe['category'] == 0),['genres','movie_title','people']]
def recommend(movie_name,recommend_number = 5):

    if movie_name in list(clean_data['movie_title']):

        movie_cluster = category_dataframe.ix[movie_name,'category']

        score = clean_data.ix[list(category_dataframe['category'] == movie_cluster),['imdb_score','movie_title']]

        sort_score = score.sort_values(['imdb_score'],ascending=[0])

        sort_score = sort_score[sort_score['movie_title'] != movie_name]

        recommend_number = min(sort_score.shape[0],recommend_number)

        recommend_movie = list(sort_score.iloc[range(recommend_number),1])

        print(recommend_movie)

    else:

        print("Can't find this movie!")
recommend('Robin Hood',10)