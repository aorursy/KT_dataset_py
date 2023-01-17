import tensorflow as tf

import os, keras

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras import backend as K

import numpy as np

import pandas as pd

import nltk, re, spacy



lyrics = pd.read_csv('../input/lyrics.csv', index_col='index').reset_index(drop=True)

lyrics.columns
lyrics['lyrics'] = lyrics['lyrics'].map(lambda l: ' '.join(str(l).split('\n'))).str.lower()
n_gram_dict = {}



def n_gram_dict_builder(s, n=5, snique=True):

    try:

        grams = []

        global n_gram_dict

        s = str(s).replace('  ',' ').replace('\'', '').replace(' ,', '').replace(' .', '').replace(' -', '')

        a = s.split(' ')

        i = len(a)

        for i2 in range(0,i,i-n):

            s1 = ' '.join(a[i2:i2+n])

            if snique:

                if s1 not in grams:

                    grams.append(s1)

                    if s1 in n_gram_dict:

                        n_gram_dict[s1] += 1

                    else:

                        n_gram_dict[s1] = 1

            else:

                grams.append(s1)

                if s1 in n_gram_dict:

                    n_gram_dict[s1] += 1

                else:

                    n_gram_dict[s1] = 1     

    except:

        grams = []

    return grams



lyrics['lyric_5_grams'] = lyrics['lyrics'].map(lambda l: n_gram_dict_builder(l, 5))
#Christmas songs are the most common lyrics shared amongs singers

[[k, n_gram_dict[k]] for k in n_gram_dict if n_gram_dict[k]>40]
artist_set = []

def curious_about(artist, grams, gram):

    if gram in grams:

        artist_set.append(artist)

    return



_ = lyrics.apply(lambda r: curious_about(r['artist'], r['lyric_5_grams'], 'have yourself a merry little'), axis=1)

print('\n'.join(sorted(set(artist_set))))
random_song = []

grams = [k for k in n_gram_dict if n_gram_dict[k]>10]

for i in range(20):

    random_song.append(np.random.choice(grams))

    

print(' '.join(random_song))
#Let tune it now with TensorFlow next

#Stay tuned (pun intended)...