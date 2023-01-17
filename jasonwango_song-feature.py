# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/songdata.csv')

df['text'] = df['text'].str.replace('\n','')

df['text'][0]
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')





df_ABBA = df.loc[df['artist']=='ABBA']

tfidf = tf.fit_transform(df_ABBA['text'])

features = tf.get_feature_names() 


indices = np.argsort(tf.idf_)[::-1]

#print(sorted_tfidf)



top_n = 25

top_features = [features[i] for i in indices[:top_n]]

print('ABBA 3-gram top features :')

print(top_features)

    
print(tfidf[0])
#type(tfidf)

ABBA_songs = tfidf.toarray()

LIMIT = 113



show_songs_features=0

ABBA_songs_list = []

for each_song in ABBA_songs:

    if show_songs_features < LIMIT:

        nonzero_features = [(features[i],x) for i,x in enumerate(each_song) if x > 0]

        #print(temp)

        nonzero_features = sorted(nonzero_features, key = lambda x : x[1],reverse=True)

        ABBA_songs_list.append(nonzero_features)

    show_songs_features += 1
song_index = 0

for each_song in ABBA_songs_list:

    print('Song: ' + df['song'][song_index])

    song_index += 1

    i = 0

    for top5_feature in each_song:

        if i < 5:

            print(top5_feature[0] + ' : ' + str(top5_feature[1]))

        i += 1

    print('------------------\n')