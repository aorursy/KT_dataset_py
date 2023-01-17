import pandas as pd
import numpy as np
import re
data = pd.read_csv('../input/anime-recommendations-database/anime.csv')
data.head()
data.isna().sum()
data['type'].unique()
data[data['episodes'] == 'Unknown']
data['genre'].unique()
data.loc[(data['type'] == 'TV') & (data['episodes'] == 'Unknown'),"episodes"] = 1
data.loc[(data['type'] == 'OVA') & (data['episodes'] == 'Unknown'),"episodes"] = 1
data.loc[(data['type'] == 'Movie') & (data['episodes'] == 'Unknown'),"episodes"] = 1
data[data['episodes'] == 'Unknown']
known_animes = {"Naruto Shippuuden":500, "One Piece":784,"Detective Conan":854, "Dragon Ball Super":86,
                "Crayon Shin chan":942, "Yu Gi Oh Arc V":148,"Shingeki no Kyojin Season 2":25,
                "Boku no Hero Academia 2nd Season":25,"Little Witch Academia TV":25}
for key,values in known_animes.items():
    data.loc[data['name'] == key, "episodes"] = values
data[data['episodes'] == 'Unknown']
data['episodes'] = data['episodes'].apply(lambda x:np.nan if x=="Unknown" else x)
data[data['episodes'] == 'Unknown']
data['episodes'].fillna(data['episodes'].median(),inplace=True)
data.isna().sum()
data['rating'].fillna(data['rating'].mean(),inplace=True)
data.isna().sum()
pd.get_dummies(data[['type']]).head()
data.dtypes
data['members'] = data['members'].astype(float)
data.dtypes
# Scaling

anime_features = pd.concat([data["genre"].str.get_dummies(sep=","),
                            pd.get_dummies(data[["type"]]),
                            data[["rating"]],data[["members"]],data["episodes"]],axis=1)
data["name"] = data["name"].map(lambda name:re.sub('[^A-Za-z0-9]+', " ", name))
anime_features.head()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
anime_features = mms.fit_transform(anime_features)
np.round(anime_features,2)
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=6,algorithm='ball_tree').fit(anime_features)
distances, indices = knn.kneighbors(anime_features)
distances
indices[67][1:]
def get_index_from_name(name):
    return data[data['name']==name].index.tolist()[0]
all_anime_names = list(data.name)
all_anime_names
def get_id_from_partial_name(partial):
    for name in all_anime_names:
        if partial in name:
            print(name,all_anime_names.index(name))
def print_similar_animes(query=None,id=None):
    if id:
        for id in indices[id][1:]:
            print(data.loc[id]["name"])
    if query:
        found_id = get_index_from_name(query)
        for id in indices[found_id][1:]:
            print(data.loc[id]["name"])
print_similar_animes(id=841)
get_id_from_partial_name('Naruto')
get_index_from_name('Naruto')
