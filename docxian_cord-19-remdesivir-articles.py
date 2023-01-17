# init

import os

import numpy as np

import pandas as pd

import time

import warnings 



import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



warnings.filterwarnings('ignore')
# load metadata

t1 = time.time()

df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

t2 = time.time()

print('Elapsed time:', t2-t1)
# define keyword

my_keyword = 'remdesivir'
def word_finder(i_word, i_text):

    found = (str(i_text).lower()).find(str(i_word).lower()) # avoid case sensitivity

    if found == -1:

        result = 0

    else:

        result = 1

    return result



# partial function for mapping

word_indicator_partial = lambda text: word_finder(my_keyword, text)

# build indicator vector (0/1) of hits

keyword_indicator = np.asarray(list(map(word_indicator_partial, df.abstract)))
# number of hits

print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())
# add index vector as additional column

df['selection'] = keyword_indicator



# select only hits from data frame

df_hits = df[df['selection']==1]
# show results

df_hits
text = " ".join(abst for abst in df_hits.abstract)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# show all abstracts

n = df_hits.shape[0]

for i in range(0,n):

    print(df_hits.title.iloc[i],":\n")

    print(df_hits.abstract.iloc[i])

    print('\n')
# make available for download

df_hits.to_csv('hits.csv')