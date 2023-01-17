import numpy as np
import pandas as pd 
import re #for immplementing regular expressions
from sklearn.feature_extraction.text import TfidfVectorizer #for implementing Tf-Idf method of Text Processing
from sklearn.metrics.pairwise import linear_kernel #for getting a similarity matrix by comparing the similarity between the data-pooints

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
podcasts=pd.read_csv("../input/all-podcast-episodes-published-in-december-2017/podcasts.csv")
print(podcasts.shape)
podcasts.head()
podcasts.info()
podcasts.language.value_counts()
podcasts=podcasts[podcasts.language == "English"]
podcasts.shape
podcasts.info()
podcasts=podcasts.dropna(subset=["description"])
podcasts.info()
podcasts=podcasts.drop_duplicates("itunes_id")
podcasts.shape
podcasts["description_#words"]=[len(x.description.split()) for _,x in podcasts.iterrows()]
podcasts["description_#words"].describe()
podcasts = podcasts[podcasts["description_#words"] > 19]
podcasts.shape
fav_podcasts = ["Earth 919: A Comic Book Podcast", "Digital India" ,"Top 5 Comics Podcast"]
fav_df=podcasts[podcasts.title.isin(fav_podcasts)]
fav_df
podcasts=podcasts[~podcasts.isin(fav_df)].sample(25000)
podcasts_df=pd.concat([fav_df,podcasts],sort=True).reset_index(drop=True)
podcasts_df=podcasts_df.dropna(subset=["description"])
print(podcasts_df.shape)
podcasts_df.head()
podcasts_df.info()
tfidf_model = TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english')
tf_idf = tfidf_model.fit_transform(podcasts_df['description'])
tf_idf
similarity_matrix = linear_kernel(tf_idf,tf_idf)
similarity_matrix
fav1=podcasts_df[podcasts_df.title == "Top 5 Comics Podcast"].index[0]
index_recom = similarity_matrix[fav1].argsort(axis=0)[-5:-1]

for i in index_recom:
    print("Score:",similarity_matrix[fav1][i],"\t Title:",podcasts_df.title[i])
    print(podcasts_df.description[i],"\n")
print("Original Description : ",podcasts_df.description[fav1])
fav2=podcasts_df[podcasts_df.title == "Digital India"].index[0]
index_recom = similarity_matrix[fav2].argsort(axis=0)[-5:-1]

for i in index_recom:
    print("Score:",similarity_matrix[fav2][i],"\t Title:",podcasts_df.title[i])
    print(podcasts_df.description[i],"\n")
print("Original Description : ",podcasts_df.description[fav2])
fav3=podcasts_df[podcasts_df.title == "Earth 919: A Comic Book Podcast"].index[0]
index_recom = similarity_matrix[fav3].argsort(axis=0)[-5:-1]

for i in index_recom:
    print("Score:",similarity_matrix[fav3][i],"\t Title:",podcasts_df.title[i])
    print(podcasts_df.description[i],"\n")
print("Original Description : ",podcasts_df.description[fav3])
fav4=podcasts_df[podcasts_df.title == "Eat Sleep Code Podcast"].index[0]
index_recom = similarity_matrix[fav4].argsort(axis=0)[-5:-1]

for i in index_recom:
    print("Score:",similarity_matrix[fav4][i],"\t Title:",podcasts_df.title[i])
    print(podcasts_df.description[i],"\n")
print("Original Description : ",podcasts_df.description[fav4])