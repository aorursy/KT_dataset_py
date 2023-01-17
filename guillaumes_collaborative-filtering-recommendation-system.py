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
dirname = '/kaggle/input/anime-recommendations-database'



rating_path = os.path.join(dirname, 'rating.csv')

anime_path = os.path.join(dirname, 'anime.csv')



rating_df = pd.read_csv(rating_path)

item_df = pd.read_csv(anime_path)
print(rating_df.shape)

print(rating_df.head())
print(item_df.shape)

print(item_df.head())
rating_df.isna().sum()
colname_mapping = {

    'anime_id': 'item_id'

}

rating_df = rating_df.rename(columns=colname_mapping)

item_df = item_df.rename(columns=colname_mapping)
print(rating_df.head())

print(item_df.head())
train = rating_df
from implicit import nearest_neighbours as nn

import scipy.sparse as sparse
sparse_item_user = sparse.csr_matrix(

    (train['rating'].astype(float),

     (train['item_id'], train['user_id'])))



model = nn.CosineRecommender()



# Calculate the confidence by multiplying it by our alpha value.

alpha_val = 15

data_conf = (sparse_item_user * alpha_val).astype('double')

model.fit(data_conf)
items_id = train.item_id.unique().tolist()

items_id[:5]
def create_item_id_to_similar(model, nb_reco, factor_similar, items_id):

    return {item_id: [(sim_item_id, sim_score)

                      for sim_item_id, sim_score in model.similar_items(

                          item_id, nb_reco * factor_similar)

                      if sim_item_id != item_id

                     ]

            for item_id in items_id

           }



item_id_to_similar = create_item_id_to_similar(

    model=model,

    nb_reco=10,

    factor_similar=5,

    items_id=items_id)
list(item_id_to_similar.items())[:2]
# visualize similar anime

max_item = 3



def print_item_features(item_id):

    current_item = item_df[item_df['item_id'] == item_id]

    print(f"id: {item_id}\nname: {current_item['name'].values[0]}\ngenre: {current_item['genre'].values[0]}")



for item_id, sim_item_ids in list(item_id_to_similar.items())[:max_item]:

    print("-----------------------------------------------")

    print_item_features(item_id=item_id)

    print("-----------------------------------------------")

    sim_item_ids = [id[0] for id in sim_item_ids]

    for counter, sim_item_id in enumerate(sim_item_ids):

        print(f"similarity rank {counter+1}")

        print_item_features(item_id=sim_item_id)

    print("-----------------------------------------------")
def get_sorted_by_values(items_id,

                         item_id_to_similar,

                         nb_reco):   

    score = {}



    for item_id in items_id:

        for sim_results in item_id_to_similar[item_id]:

            score[sim_results[0]] = score.get(sim_results[0], 0) + sim_results[1]

  

    # order the dictionary to identify the most similar animes

    sorted_by_value = sorted(score.items(), key=lambda kv: kv[1], reverse=True)

    return sorted_by_value
sorted_by_value = get_sorted_by_values(items_id=items_id[:10],

                     item_id_to_similar=item_id_to_similar,

                     nb_reco=5

                    )
sorted_by_value
def user_to_visited_item_id_dict(train,

                                 user_list):

    return train.groupby('user_id')['item_id'].apply(lambda g: g.values

                                                  .tolist()).to_dict()
user_to_visited_item_id_dict = user_to_visited_item_id_dict(train=train,

                                                            user_list=train.user_id.unique().tolist())
def print_visited_item(user_id):

    visited_item_list = user_to_visited_item_id_dict[user_id]

    print(f"For user_id: {user_id}, visited animes (count:{len(visited_item_list)}) are:")

    for counter, item_id in enumerate(visited_item_list):

        print(f"visited anime {counter+1}")

        print_item_features(item_id=item_id)
print_visited_item(user_id=train.user_id.values[0])
def compute_recommendation(user_id, user_to_visited_item_id_dict, item_id_to_similar, nb_reco=5):

    visited_item_list = user_to_visited_item_id_dict[user_id]

    sorted_by_values = get_sorted_by_values(

        items_id=visited_item_list,

        item_id_to_similar=item_id_to_similar,

        nb_reco=nb_reco

    )

    return [id_score[0] for id_score in sorted_by_values][:nb_reco]



def print_user_recommendation(user_id, user_to_visited_item_id_dict, item_id_to_similar, nb_reco=5):

    recommendation_list = compute_recommendation(

        user_id=user_id, 

        user_to_visited_item_id_dict=user_to_visited_item_id_dict, 

        item_id_to_similar=item_id_to_similar, 

        nb_reco=nb_reco)

    

    for counter, item_id in enumerate(recommendation_list):

        print(f"recommended item {counter+1}")

        print_item_features(item_id=item_id)
compute_recommendation(

    user_id=train.user_id.values[0],

    user_to_visited_item_id_dict=user_to_visited_item_id_dict,

    item_id_to_similar=item_id_to_similar,

    nb_reco=5)
print_user_recommendation(

    user_id=train.user_id.values[0],

    user_to_visited_item_id_dict=user_to_visited_item_id_dict,

    item_id_to_similar=item_id_to_similar,

    nb_reco=5)
def print_visited_anime_and_recommendation(

    user_id,

    user_to_visited_item_id_dict=user_to_visited_item_id_dict, 

    item_id_to_similar=item_id_to_similar, 

    nb_reco=5

):

    

    print_visited_item(user_id=user_id)

    

    print_user_recommendation(

        user_id=user_id,

        user_to_visited_item_id_dict=user_to_visited_item_id_dict,

        item_id_to_similar=item_id_to_similar,

        nb_reco=nb_reco

    )





import random

def get_random_user(users):

    return random.choice(users)
print_visited_anime_and_recommendation(

    user_id=get_random_user(users=train.user_id.unique().tolist())

)