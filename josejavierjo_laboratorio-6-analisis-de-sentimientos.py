# Matematica

import pandas as pd

import numpy as np



# Visualización

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

color = sns.color_palette()

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import plotly.tools as tls



from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



from wordcloud import WordCloud, STOPWORDS



import warnings

warnings.filterwarnings('ignore')



import os

os.listdir("../input/grammar-and-online-product-reviews")
df=pd.read_csv('../input/grammar-and-online-product-reviews/GrammarandProductReviews.csv')

df.head()
# Lineas nulas

df.isnull().sum()
# Eliminar nulos

df = df.dropna(subset=['reviews.text'])
df['reviews.text'] = df['reviews.text'].apply(lambda line: line.lower())
import string

punc_ext = string.punctuation + '¡¿'

def remove_punctuation(text):

    return text.translate(text.maketrans('', '', punc_ext))
remove_punctuation('¡hola! TIO!?')
df['reviews.text'] = df['reviews.text'].apply(lambda line: remove_punctuation(line))
import re

def remove_url(text):

    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
df['reviews.text'] = df['reviews.text'].apply(lambda line: remove_url(line))
import emoji

df['reviews.text'] = df['reviews.text'].apply(lambda line: emoji.demojize(line))
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
sns.set(style="darkgrid")

sns.countplot(df['reviews.rating'])
r1 = df.ix[df['reviews.rating']==1, ['reviews.text']]

r2 = df.ix[df['reviews.rating']==2, ['reviews.text']]

r3 = df.ix[df['reviews.rating']==3, ['reviews.text']]

r4 = df.ix[df['reviews.rating']==4, ['reviews.text']]

r5 = df.ix[df['reviews.rating']==5, ['reviews.text']]
stopwords = set(STOPWORDS)



def most_used_words(data):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    plt.imshow(wordcloud)

    plt.show()
most_used_words(r1)
most_used_words(r2)
most_used_words(r3)
most_used_words(r4)
most_used_words(r5)
df['reviews_length'] = df['reviews.text'].apply(len)

g = sns.FacetGrid(df,col='reviews.rating',size=5)

g.map(plt.hist,'reviews_length', range=(0, 1200))
corr = df.corr()

f, ax = plt.subplots(figsize=(10, 5))

sns.heatmap(corr, cbar=True, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
def classifyRating(rate):

    if rate < 3:

        return 'Bad'

    elif rate == 3:

        return 'Neutral'

    else:

        return 'Good'
df['ReviewType'] = df['reviews.rating'].apply(lambda rate: classifyRating(rate))
def classifyRating2(rate):

    if rate < 3:

        return '0'

    elif rate == 3:

        return '1'

    else:

        return '2'
df['numberRate'] = df['reviews.rating'] < 4
buenas = {}

malas = {}

    

for review, classType in zip(df['reviews.text'], df['ReviewType']):

    text = review.split(' ')

    for word in text:

        if word not in stop_words and word != '':

            if classType == 'Good':

                counter = buenas.get(word)

                if counter:

                    buenas[word] = counter + 1

                else:

                    buenas[word] = 1

            elif classType == 'Bad':

                counter = malas.get(word)

                if counter:

                    malas[word] = counter + 1

                else:

                    malas[word] = 1

                
import operator

best_good = sorted(buenas.items(), key=operator.itemgetter(1))

best_good.reverse()

best_good
best_bad = sorted(malas.items(), key=operator.itemgetter(1))

best_bad.reverse()

best_bad
def return_next(modelo, texto):

    return diccionario[:6]
from keras.layers import Dense, Input, Flatten

from keras.layers import GlobalAveragePooling1D, Embedding

from keras.models import Model
np.random.seed(32)



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.manifold import TSNE



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout

from keras.utils.np_utils import to_categorical
train_text, test_text, train_y, test_y = train_test_split(df['reviews.text'],df['numberRate'],test_size = 0.3)
MAX_NB_WORDS = 20000



texts_train = train_text.astype(str)

texts_test = test_text.astype(str)



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)

tokenizer.fit_on_texts(texts_train)

sequences = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())

" ".join([index_to_word[i] for i in sequences[0]])
MAX_SEQUENCE_LENGTH = 150



x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
y_train = train_y

y_test = test_y



y_train = to_categorical(np.asarray(y_train))

print('Shape of label tensor:', y_train.shape)
from keras.layers import Dense, Input, Flatten

from keras.layers import GlobalAveragePooling1D, Embedding

from keras.models import Model



EMBEDDING_DIM = 50

N_CLASSES = 2



# input: a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

embedded_sequences = embedding_layer(sequence_input)



average = GlobalAveragePooling1D()(embedded_sequences)

predictions = Dense(N_CLASSES, activation='softmax')(average)



model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam', metrics=['acc'])
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)

predictions = Dense(2, activation='softmax')(x)





model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.1,

          nb_epoch=2, batch_size=128)
output_test = model.predict(x_test)

print("test auc:", roc_auc_score(y_test,output_test[:,1]))
def predictor(modelo, texto):

    return return_next(modelo, texto)

    
predictor(model, 'this is a test of a bad bad bad bad bad review i hated this product a lot it sucked')