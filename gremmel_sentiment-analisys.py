# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
amazon_df = pd.read_csv("../input/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt", 
                        delimiter='\t', 
                        header=None, 
                        names=['review', 'sentiment'])

imdb_df = pd.read_csv("../input/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt", 
                        delimiter='\t', 
                        header=None, 
                        names=['review', 'sentiment'])

yelp_df = pd.read_csv("../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt", 
                        delimiter='\t', 
                        header=None, 
                        names=['review', 'sentiment'])
amazon_df.head()
imdb_df.head()
yelp_df.head()
all_df = pd.concat([amazon_df, imdb_df, yelp_df])
all_df.reset_index(drop=True, inplace=True)

reviews = all_df['review'].get_values()
sentiment = all_df['sentiment'].get_values()
import  re
from nltk.corpus import stopwords
import unidecode
print(reviews[1])
a = unidecode.unidecode(reviews[1])
a = re.sub('[^a-zA-Z ]', '', reviews[1])
print(a)
print(a.lower())

clean_reviews = []

for review in reviews:
    # removendo acentos
    a = unidecode.unidecode(review)
    # removendo nao letras
    a = re.sub('[^a-zA-Z ]', '', a)
    # lowercase
    a = a.lower()
    
    # faz o split pelo espaco
    a = a.split()

    # removendo stop words
    novo_a = []
    for word in a:
        if not word in stopwords.words('english'):
            novo_a.append(word)
    
    clean_reviews.append(novo_a)

reviews[0]
len(clean_reviews)
from collections import Counter
word_freq = Counter()
for review in clean_reviews:
    for word in review:
        word_freq[word] += 1
top3000_words = word_freq.most_common(3000)
word_to_id = {}
# palavras vao receber o valor do seu rank
idx = 3000
for t in top3000_words:
    word_to_id[t[0]] = idx
    idx -= 1

# palavras que nao foram rankeadas vao receber valor 1
for word in word_freq.keys():
    if not word in word_to_id:
        word_to_id[word] = 1
data = []
for review in clean_reviews:
    aux = []
    for word in review:
        aux.append(word_to_id[word])
    
    data.append(aux)
from keras.preprocessing.sequence import pad_sequences
data = pad_sequences(data, maxlen=100, padding='post')
from keras.layers import *
from keras.models import Model
input_node = Input(shape=(None, 100))
embedding_layer = Embedding(input_dim=3001, output_dim=32)(input_node)
reshape = Reshape((100, 32))(embedding_layer)
lstm = LSTM(100)(reshape)
drop = Dropout(0.2)(lstm)
output_node = Dense(1, activation='sigmoid')(drop)

model = Model(input_node, output_node)
model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy'])
data = data.reshape(len(data), 1, 100)
model.fit(data, sentiment, epochs=1)