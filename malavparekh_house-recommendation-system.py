import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

import os

os.listdir('../input/')
prop = pd.read_json('../input/property-data/Property.json')

user = pd.read_json('../input/property-data/User.json')

prop.shape, user.shape
user.head()
prop.head()
prop['price'].head()
def removo_punc(row):

    return float(''.join(str(row).split(','))[1:])



prop['price'] = prop['price'].astype(str).map(removo_punc)

prop['price']
sns.countplot(prop['bedroom'])
sns.countplot(prop['bathroom'])
sns.scatterplot(prop['latitude'], prop['longitude'], alpha=0.7)
prop['tags'].head()
def remove_space(row):

    return ['_'.join(i.split(' ')) for i in row]



prop['tags'] = prop['tags'].map(remove_space)



set1 = set()

for i in prop['tags']:

    for j in i:

        set1.add(j)

print(len(set1))

print(set1)
def tag_one_hot(row):

    dict1 = {'Bathrooms':0, 'Bedrooms':0,'Living_rooms':0, 'Location':0, 'Picture':0, 'Price':0, 'Schools':0, 'Size_of_home':0}

    for i in row:

        dict1[i] = 1

    return [i for i in dict1.values()]

tags = prop['tags'].map(tag_one_hot)



tag_cols = ['Bathrooms', 'Bedrooms','Living_rooms', 'Location', 'Picture', 'Price', 'Schools','Size_of_home']

tag_data = pd.DataFrame(tags.tolist(), columns=tag_cols)

tag_data.head()
prop = prop.drop(['tags','_id','picture','address'],axis=1)



final_prop = pd.concat((prop, tag_data),axis=1)

final_prop.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

final_prop = scaler.fit_transform(final_prop)
from sklearn.metrics.pairwise import cosine_similarity



cosine_simi = cosine_similarity(final_prop)

plt.figure(figsize=(20,20))

sns.heatmap(cosine_simi)
cosine_simi = pd.DataFrame(cosine_simi, columns = [i for i in range(100)], index = [i for i in range(100)])

cosine_simi.head()
user.head()
def recommendations(row):

    props = {}

    for i in row:

        props[cosine_simi[i].sort_values(ascending=False).index[1]] = cosine_simi[i].sort_values(ascending=False)[1]

    return [i for i,j in props.items() if (j>0.5) & (i not in row)]
user['Recommendation'] = user.userSaveHomes.map(recommendations)

user.head()