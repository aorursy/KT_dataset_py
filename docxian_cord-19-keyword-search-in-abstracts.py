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

my_keyword = 'main protease'
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
# look at an example: metadata first

df_hits.iloc[0]
# look at an example: the abstract itself

df_hits.abstract.iloc[0]
# define keyword

my_keyword2 = 'vaccine'
# partial function for mapping

word_indicator_partial2 = lambda text: word_finder(my_keyword2, text)

# build indicator vector (0/1) of hits

keyword_indicator2 = np.asarray(list(map(word_indicator_partial2, df_hits.abstract)))
# number of hits

print('Number of hits for keywords <', my_keyword, '> + <', my_keyword2, '> : ', keyword_indicator2.sum())
# add index vector as additional column

df_hits['selection'] = keyword_indicator2



# select only hits from data frame

df_hits2 = df_hits[df_hits['selection']==1]
df_hits2
# look at an example: metadata first

df_hits2.iloc[0]
# look at an example: the abstract itself

df_hits2.abstract.iloc[0]
# save selection to CSV file for further evaluations

df_hits2.to_csv('selection.csv')
text = " ".join(abst for abst in df_hits2.abstract)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()