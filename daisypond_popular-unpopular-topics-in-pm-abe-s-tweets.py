import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import glob

import sklearn as sk

from sklearn.feature_extraction import stop_words

from wordcloud import WordCloud
df = pd.read_csv('../input/Shinzo Abe Tweet 20171024 - Tweet.csv')

df.head(5)
df['Replies'] = df['Profile Tweet 1'].replace('[^0-9]', '', regex=True).astype(np.int64)

df['Retweets'] = df['Profile Tweet 2'].replace('[^0-9]', '', regex=True).astype(np.int64)

df['Likes'] = df['Profile Tweet 3'].replace('[^0-9]', '', regex=True).astype(np.int64)
df.head(5)
att = ['Replies','Retweets', 'Likes']

pd.plotting.scatter_matrix(df[att])
df.describe()
data = df[['Retweets','Likes']]

data.corr(method='pearson')
data = df[['Replies','Retweets']]

data.corr(method='pearson')
data = df[['Likes','Replies']]

data.corr(method='pearson')
like_mean = df['Likes'].mean()



df_popular     = df.query('Likes > '+ str(like_mean)) 

df_unpopular = df.query('Likes <= '+ str(like_mean)) 

def add_words(word_set,text):

    

    words = text.split(' ')

    word_set = word_set | set(words)

    return word_set



def delete_words(words, text):

    for w in words:

        text= text.replace(" " + w + " ", ' ')

        text = text.replace('pictwittercom','')

    return text



stop = stop_words.ENGLISH_STOP_WORDS

text_unpop = df_unpopular['English Translation'].replace('[¥.¥,¥!¥?]', '', regex=True)

text_pop     = df_popular['English Translation'].replace('[¥.¥,¥!¥?]', '', regex=True)



words_unpop = set()

words_pop    = set()



unpop_text =""

pop_text =""



for w in text_unpop:

    words_unpop = add_words(words_unpop, w)

    unpop_text = unpop_text + " " + w



for w in text_pop:

    words_pop = add_words(words_pop, w)

    pop_text = pop_text + " " + w

    

unpop_text = delete_words(words_pop, unpop_text)

unpop_text = delete_words(stop, unpop_text)

pop_text = delete_words(words_unpop , pop_text)

pop_text = delete_words(stop, pop_text)
wordcloud = WordCloud().generate(unpop_text)

plt.figure(figsize=(15, 15), dpi=50)

plt.imshow(wordcloud, interpolation='bilinear')



plt.show()
wordcloud = WordCloud().generate(pop_text)

plt.figure(figsize=(15, 15), dpi=50)

plt.imshow(wordcloud, interpolation='bilinear')

plt.show()