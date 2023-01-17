import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer # IFIDF vectorizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Reading data

dataset = pd.read_csv('../input/songdata.csv')



# Transform into TFIDF

corpus = dataset['text'].tolist()

tfidf = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, stop_words='english')

tfidf.fit_transform(corpus)



# 

for artist, data in dataset.groupby('artist'):

    feat = np.sum(tfidf.transform(data['text']).toarray(), axis=0)

    sort_idx = np.argsort(feat.flatten())[::-1] # Sort descending

    fav = [tfidf.get_feature_names()[idx] for idx in sort_idx.tolist()[:10]]

    print(artist + "'s Fav words:")

    for w in fav:

        print('\t' + w)

    print()

    