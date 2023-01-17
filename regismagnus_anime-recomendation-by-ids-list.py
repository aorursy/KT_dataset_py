import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
anime_data = pd.read_csv('../input/anime-recommendations-database/anime.csv', index_col='anime_id')
#change Unknown as NaN

anime_data.replace("Unknown", np.nan, inplace=True)

#remove all NaN values

anime_data.dropna(inplace=True)



#convert episodes to numeric

anime_data['episodes']=pd.to_numeric(anime_data['episodes'])



#capture all genres

genres=[]

for group_genres in anime_data['genre']:

    if not pd.isna(group_genres):

        split_genres=group_genres.split(',')

        for genre in split_genres:

            genre=genre.strip()

            if genres.count(genre)==0:

                genres.append(genre)

#create cols by genres

for genre in genres:

    anime_data['genre_'+genre]=[False for i in range(anime_data.shape[0])]

    

def set_genre(row):

    '''set True in the col of relative genre'''

    if not pd.isna(row['genre']):

        genres=row['genre'].split(',')

        for genre in genres:

            genre=genre.strip()

            row['genre_'+genre] = True

    return row

#set genre True in correct places

anime_data = anime_data.apply(set_genre, axis=1)



encoder=LabelEncoder()

# Apply the label encoder to type

anime_data['type_n'] = anime_data[['type']].apply(encoder.fit_transform)
#list of anime id that already watched

list_id=[5114, 11061, 4181, 263, 1, 30276, 33, 57, 72, 205, 19, 245]



#use all features. Members not be used because is natural the new animes not have lot of members

X_features=np.concatenate([['type_n', 'episodes', 'rating'], ['genre_' + genre for genre in genres]])



#remove animes in the list id to not recommend they again

data_train=anime_data[~anime_data.index.isin(list_id)]



model_1=DecisionTreeClassifier(random_state=42)

model_1.fit(data_train[X_features], data_train.index)
#predict and remove duplicates

predics=list(dict.fromkeys(model_1.predict(anime_data[anime_data.index.isin(list_id)][X_features])))
#show the result in order of members and rating

anime_data[anime_data.index.isin(predics)].sort_values(['members', 'rating'], ascending=False)[['name', 'rating']]