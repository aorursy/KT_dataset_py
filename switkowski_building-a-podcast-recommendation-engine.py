import numpy as np
import pandas as pd 
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
podcasts = pd.read_csv('../input/podcasts.csv')
podcasts.head()
podcasts.info()
(podcasts
 .language
 .value_counts()
 .to_frame()
 .head(10))
podcasts = podcasts[podcasts.language == 'English']
podcasts = podcasts.dropna(subset=['description'])
podcasts = podcasts.drop_duplicates('itunes_id')
sum(podcasts.description.isnull())
podcasts['description_length'] = [len(x.description.split()) for _, x in podcasts.iterrows()]
podcasts['description_length'].describe()
podcasts = podcasts[podcasts.description_length >= 20]
favorite_podcasts = ['The MFCEO Project', 'Up and Vanished', 'Lore']
favorites = podcasts[podcasts.title.isin(favorite_podcasts)]
favorites
podcasts = podcasts[~podcasts.isin(favorites)].sample(15000)
data = pd.concat([podcasts, favorites], sort = True).reset_index(drop = True)
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = "english")
tf_idf = tf.fit_transform(data['description'])
tf_idf
similarity = linear_kernel(tf_idf, tf_idf)
similarity
x = data[data.title == 'Up and Vanished'].index[0]
similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
for i in similar_idx:
    print(similarity[x][i], '-', data.title[i], '-', data.description[i], '\n')
print('Original - ' + data.description[x])
x = data[data.title == 'Lore'].index[0]
similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
for i in similar_idx:
    print(similarity[x][i], '-', data.title[i], '-', data.description[i], '\n')
print('Original - ' + data.description[x])