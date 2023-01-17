# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.tokenize import WordPunctTokenizer

from nltk.corpus import stopwords

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.collocations import TrigramCollocationFinder

from nltk.metrics import TrigramAssocMeasures

from nltk.tokenize import MWETokenizer

from nltk.stem.snowball import SnowballStemmer

import string

import re



bigram_measures = BigramAssocMeasures()

trigram_measures = TrigramAssocMeasures()



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation



import os

print(os.listdir("../input"))



from wordcloud import WordCloud

import matplotlib.pyplot as plt 

%matplotlib inline  



import multiprocessing, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)   

logging.getLogger("py4j").setLevel(logging.ERROR)



from nltk.data import find

from gensim.models import word2vec

import gensim



import spacy

nlp = spacy.load('en_core_web_lg')

# nlp = spacy.load('../input/spacyen-vectors-web-lg/spacy-en_vectors_web_lg/en_vectors_web_lg/')



word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras.layers.embeddings import Embedding

from keras.initializers import Constant
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
path = '../input/'

data = pd.read_json(path+'news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)

print(data['is_sarcastic'].value_counts())

data.head()
[print(item, '\n') for item in data['headline'][3:7]]

print()
obj = data['headline'][4]

print(obj)

doc = nlp(obj)



for token in doc:

    print(token.string, token.pos_, token.tag_, token.dep_)
def clean_text(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)

    ## Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    return text

# apply the above function to df['text']

data['headline_text'] = data['headline'].map(lambda x: clean_text(x))
#vectorize

list_news = data['headline_text'].tolist()

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(list_news)

sequences = tokenizer_obj.texts_to_sequences(list_news)



max_length = max([len(s.split()) for s in list_news])

print('Maximum length of sentences %s' %max_length)

#pad sequence

word_index = tokenizer_obj.word_index

print('number of unique words', len(word_index))



review_pad = pad_sequences(sequences, maxlen=max_length)
embedding_index = {}

for item in w2v_model.wv.vocab:

    embedding_index[item.lower()] = w2v_model[item]



EMBEDDING_DIM = embedding_index['adam'].shape[0]

#initializing with zero vector for Unknown word

embedding_index['<UNK>'] = np.zeros(EMBEDDING_DIM)



num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))



for word, ind in word_index.items():

    if ind>num_words:

        continue

    embedding_vector = embedding_index.get(word, embedding_index['<UNK>'])

    if embedding_vector is not None:

        embedding_matrix[ind] = embedding_vector

print(embedding_matrix.shape)
model = Sequential()

embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,

                           embeddings_initializer=Constant(embedding_matrix),

                           input_length=max_length,

                           trainable=False,

                           name='Embedding')

model.add(embedding_layer)

model.add(LSTM(units=max_length, dropout=0.2, name='lstm' ))

model.add(Dense(units=1, activation='sigmoid', name='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
VALIDATION_SPLIT = 0.2

indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)



review_pad = review_pad[indices]

sarcasm_flag = np.array(data['is_sarcastic'].tolist())[indices]



no_validation_sample = int(VALIDATION_SPLIT*review_pad.shape[0])



X_train_pad = review_pad[:-no_validation_sample]

y_train = sarcasm_flag[:-no_validation_sample]



x_test_pad = review_pad[-no_validation_sample:]

y_test = sarcasm_flag[-no_validation_sample:]
print('Let\'s Train...')

model.fit(X_train_pad, y_train, batch_size=64, epochs=32, validation_data=(x_test_pad, y_test), verbose=2)
#finding bias vector 

#person vector with top 5000 baby names

person_names = pd.read_csv('../input/us-baby-names/NationalNames.csv')['Name'].head(5000).tolist()

person_vector = []

for item in person_names:

    try:

        person_vector.append(embedding_matrix[item.lower()].tolist())

    except:

        pass

person_matrice = np.array(person_vector)

print('Person matched', person_matrice.shape[0])

embedding_matrix['[NAME]'] = np.mean(person_matrice, axis=0)



#location vector with city names

location_names = pd.read_csv('../input/store-locations/directory.csv')['City'].unique().tolist()

location_vector = []

for item in location_names:

    try:

        location_vector.append(embedding_matrix[item.lower()].tolist())

    except:

        pass

location_matrice = np.array(location_vector)

print('Location matched', location_matrice.shape[0])

embedding_matrix['[PLACE]'] = np.mean(location_matrice, axis=0)