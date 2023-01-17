import numpy as np
import pandas as pd
df_train = pd.read_csv("../input/u.data.csv", names=['user_id', 'item_id', 'ranking', 'time'], sep='\t')
df_train['time'] = pd.to_datetime(df_train['time'],unit='s')
df_train = df_train.sort_values(by='time')
df_train = df_train[df_train['ranking'] > 3]
df_train.head()
names=['item_id', 'movie title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western' ]
df_items = pd.read_csv("../input/u.item", names= names, sep='|', encoding = 'ISO-8859-1')
df_items = df_items.filter(['item_id', 'movie title', 'Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western'])
df_items.head()
df_train_2 = pd.merge(df_train, df_items, on='item_id')
df_train_2[df_train_2['user_id'] == 914]
l = ['Action', 'Adventure', 'Animation', 
    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western']
    
def f(row):
    sum = 0
    for i in l:
        sum = sum+row[i] 
    return sum
    
df_train_2['Total'] = df_train_2.apply(f, axis=1)
df_train_2.head()
df_items['Total'] = df_items.apply(f, axis=1)
df_items['Total'].describe()
def get_features_movie(item_id):
    l = ['Action', 'Adventure', 'Animation', 
    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western']
    
    features = []
    temp = df_items[df_items['item_id'] == item_id]
    for i in l:
        if temp.iloc[0][i]:
            features = features + [i]
    
    return features

def convert_to_list(item_id):
    return " ".join([str(x) for x in get_features_movie(item_id)[:1]])

train_watched = pd.DataFrame(columns=['user_id', 'watched'])

for index, user_id in enumerate(range(min(df_train_2['user_id']), max(df_train_2['user_id']))):
    d = df_train_2[df_train_2['user_id'] == user_id].filter(['item_id'])
    l = d['item_id'].tolist()
    l_to_string = " ".join([convert_to_list(x)+" "+str(x) for x in l])
    train_watched.loc[index] = [user_id, l_to_string]
train_watched.head()
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
list_doc = []

for row in train_watched.to_dict(orient='record'):
    list_doc.append(str(row['watched']).strip().split(' '))
model = Word2Vec(list_doc, window=5, min_count=1, workers=4)
def most_similar(item_id_or_genre):
    try:
        print("Similar of "+df_items[df_items['item_id'] == int(item_id_or_genre)].iloc[0]['movie title'])
    except:
        print("Similar of "+item_id_or_genre)
    return [(x, df_items[df_items['item_id'] == int(x[0])].iloc[0]['movie title']) for x in model.wv.most_similar(item_id_or_genre)]
most_similar('Action')
most_similar('Horror')
most_similar('226')
df_train_2[df_train_2['user_id'] == 914].filter(['item_id', 'movie title']+l)
def create_avg_user_vector(user_id):
    item_id_list = df_train_2[df_train_2['user_id'] == user_id]['item_id'].tolist()
    vector_item_id_list = [model.wv[str(x)] for x in item_id_list]
    return np.average(vector_item_id_list, axis=0)

def most_similar_by_vector(vector):
    return [(x, df_items[df_items['item_id'] == int(x[0])].iloc[0]['movie title']) for x in model.wv.similar_by_vector(vector)]

most_similar_by_vector(create_avg_user_vector(914))