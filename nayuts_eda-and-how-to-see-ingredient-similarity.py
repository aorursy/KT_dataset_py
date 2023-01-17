import collections

import json

import os



import folium

import matplotlib.pyplot as plt

import nltk

import numpy as np

import re

import pandas as pd

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity

import umap

from wordcloud import WordCloud
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!ls ../input/indian-food-101
df = pd.read_csv("/kaggle/input/indian-food-101/indian_food.csv")
df.head()
df.shape
df.info()
print("There are ", len(set(df['name'])), "dish")
g_diet = sns.countplot(data=df, x="diet", order = df['diet'].value_counts().index)

g_diet.set_title("diet countplot")
df_diet =  df[["diet"]].copy()

df_diet["count"] = 1

df_diet = df_diet.groupby("diet").count()

df_diet.head()
df_diet.plot.pie(y="count")
g_flavor_profile = sns.countplot(data=df, x="flavor_profile", order = df['flavor_profile'].value_counts().index)

g_flavor_profile.set_title("flavor_profile countplot")
df_flavor_profile =  df[["flavor_profile"]].copy()

df_flavor_profile["count"] = 1

df_flavor_profile = df_flavor_profile.groupby("flavor_profile").count()

df_flavor_profile.plot.pie(y="count")
g_course = sns.countplot(data=df, x="course", order = df['course'].value_counts().index)

g_course.set_title("course countplot")
g_region = sns.countplot(data=df, x="region", order = df['region'].value_counts().index)

g_region.set_title("region countplot")
g_region = sns.countplot(data=df, x="region", hue="flavor_profile", order = df['region'].value_counts().index)

g_region.set_title("region countplot")
plt.figure(figsize=(20, 10))

g_state = sns.countplot(data=df, x="state",order = df['state'].value_counts().index)

g_state.set_xticklabels(g_state.get_xticklabels(), rotation=45)

g_state.set_title("state countplot")
g_prep_time = sns.distplot(df["prep_time"])

g_prep_time.set_title("prep_time countplot")
g_cook_time = sns.distplot(df["cook_time"])

g_cook_time.set_title("cook_time countplot")
all_words = []

for i in range(len(df)):

    txt =  df["ingredients"][i]

    #txt =  txt.replace(', ', ',').lower()

    #all_words += [ word for word in re.split('[,.]',txt) ]

    all_words += [word.lower() for word in nltk.word_tokenize(txt) if not word in ['.', ',']]

    

word_freq = collections.Counter(all_words)

W = WordCloud().fit_words(word_freq)
plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(W)

plt.axis('off')

plt.show()
words = np.array(list(word_freq.keys()))

words
def gen_ingredients_vector(ingredients):

    ingredients_vec = np.zeros(words.shape)

    ingredients = set([word.lower() for word in nltk.word_tokenize(ingredients) if not word in ['.', ',']])

    for ingredient in ingredients:

        idx = np.where(words == ingredient)

        ingredients_vec[idx] = 1

    return ingredients_vec.tolist()



df["ingredients_vec"] = df["ingredients"].map(gen_ingredients_vector)



df.head()
# To input heatmap, I made ingredients_vecs.

ingredients_vecs = []

for i in range(len(df)):

    ingredients_vecs.append(df["ingredients_vec"][i])

    

ingredients_vecs = np.array(ingredients_vecs)
ingredients_vecs
cos_matrix = cosine_similarity(ingredients_vecs, ingredients_vecs)
plt.figure(figsize=(20, 20))

ax = sns.heatmap(cos_matrix)

ax.set_title("cosine_similarity of ingredients_vectors")
df.iloc[[10,12]]
cosine_similarity([ingredients_vecs[10]], [ingredients_vecs[12]])
df.iloc[[10,56]]
cosine_similarity([ingredients_vecs[10]], [ingredients_vecs[56]])
kmeans = KMeans(n_clusters=5, random_state=0).fit(ingredients_vecs)
reducer = umap.UMAP()

embedding = reducer.fit_transform(ingredients_vecs)
plt.scatter(

    embedding[:, 0],

    embedding[:, 1],

    c=kmeans.labels_)

plt.gca().set_aspect('equal', 'datalim')

plt.title('UMAP projection of ingredients vectors', fontsize=15)
def get_cosine_similarity_heatmap(df,state=None, region=None):

    """

    Visualize cosine similarity heatmap of ingredients vector.

    And return filtered dataframe.

    """

    

    if state==None and region==None:

        df_filtered = df

    elif state!=None:

        df_filtered = df[df["state"]==state]

    elif region!=None:

        df_filtered = df[df["region"]==region]

    else:

        df_filtered = df[df["state"]==state & df["region"]==region]

    

    #ingredients_vecs = []

    

    ingredients_vecs = [vec for vec in df_filtered["ingredients_vec"]]

    #for i in range(len(df_filtered)):

    #    ingredients_vecs.append(df_filtered["ingredients_vec"][i])

    

    ingredients_vecs = np.array(ingredients_vecs)

    cos_matrix = cosine_similarity(ingredients_vecs, ingredients_vecs)

    

    plt.figure(figsize=(15, 15))

    ax = sns.heatmap(cos_matrix)

    ax.set_title(f"cosine_similarity of ingredients_vectors in {state} and {region}")

    

    return df_filtered.reset_index()
df_WestBengal_East = get_cosine_similarity_heatmap(df, "West Bengal", "East")
df_WestBengal_East.iloc[[5,8]]