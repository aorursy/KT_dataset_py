import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

display(df.info())

display(df.head())
df = df.dropna(subset = ['Genre', 'Publisher']).reset_index()

df['Criteria'] = df['Genre'].str.cat(df[['Platform', 'Publisher']], sep = ' ')
display(df.isnull().sum())

display(print(df.head()))
# count genre

from sklearn.feature_extraction.text import CountVectorizer

model = CountVectorizer(

    tokenizer = lambda i: i.split(' '),    # => cari split karakter yg unik

    analyzer = 'word'

)

matrix_genre = model.fit_transform(df['Criteria'])

tipe_genre = model.get_feature_names()

jumlah_genre = len(tipe_genre)

event_genre = matrix_genre.toarray()
# cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

score = cosine_similarity(matrix_genre)
# test model

saya_suka = 'Suikoden'

display(df[df['Name'] == saya_suka])



# take the index from saya_suka

index_suka = df[df['Name'] == saya_suka].index.values[0]



# list all games + cosine similarity score

all_games = list(enumerate(score[index_suka]))



# show 5 first datas, sorted by index

game_sama = sorted(

    all_games,

    key = lambda i: i[1],

    reverse = True

)
# for i in game_sama[:5]:

#     print(

#         df.iloc[i[0]]['Name'],

#         df.iloc[i[0]]['Platform'],

#         df.iloc[i[0]]['Genre'],

#         df.iloc[i[0]]['Publisher']

#         )
# list all games filter by cosine similarity score > 80%

game_80up = []

for i in game_sama:

    if i[1] > 0.8:

        game_80up.append(i)



# show 5 datas randomly, where cosine similarity score > 50%

import random

rekomendasi = random.choices(game_80up, k = 5)



for i in rekomendasi:

    print(

        df.iloc[i[0]]['Name'],

        df.iloc[i[0]]['Platform'],

        df.iloc[i[0]]['Genre'],

        df.iloc[i[0]]['Publisher']

    )