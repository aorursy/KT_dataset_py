# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import string

import re

import glob

import multiprocessing

import time



from gensim.models.doc2vec import Doc2Vec, TaggedDocument



from keras.backend import floatx

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer, sent_tokenize

from nltk.stem import WordNetLemmatizer



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding



from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator

%matplotlib inline
df = pd.read_csv('/kaggle/input/precily-test/bbc-text.csv')
bbc_news_categories = {

    'business': 0,

    'entertainment': 1,

    'politics': 2,

    'sport': 3,

    'tech': 4

}
bbc_news_X = []

bbc_news_Y = []



stop_words = set(stopwords.words('english'))

regexp_tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

wordnet_lemmatizer = WordNetLemmatizer()
sport = df[df['category'] == 'sport']

business = df[df['category'] == 'business']

entertainment = df[df['category'] == 'entertainment']

politics = df[df['category'] == 'politics']

tech = df[df['category'] == 'tech']
text_all = " ".join(review for review in df.text)

text_sport = " ".join(review for review in sport.text)

text_buss = " ".join(review for review in business.text)

text_ent = " ".join(review for review in entertainment.text)

text_pol = " ".join(review for review in politics.text)

text_tech = " ".join(review for review in tech.text)
fig, ax = plt.subplots(6, 1, figsize  = (30,30))

wordcloud_ALL = WordCloud(max_font_size=50, max_words=1000, background_color="white").generate(text_all)

wordcloud_sport = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_sport)

wordcloud_buss = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_buss)

wordcloud_ent = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_ent)

wordcloud_pol = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_pol)

wordcloud_tech = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_tech)



ax[0].imshow(wordcloud_ALL, interpolation='bilinear')

ax[0].set_title('All text', fontsize=30)

ax[0].axis('off')

ax[1].imshow(wordcloud_sport, interpolation='bilinear')

ax[1].set_title('Sport', fontsize=30)

ax[1].axis('off')

ax[2].imshow(wordcloud_buss, interpolation='bilinear')

ax[2].set_title('Bussiness', fontsize=30)

ax[2].axis('off')

ax[3].imshow(wordcloud_ent, interpolation='bilinear')

ax[3].set_title('Entertainment', fontsize=30)

ax[3].axis('off')

ax[4].imshow(wordcloud_pol, interpolation='bilinear')

ax[4].set_title('Politics', fontsize=30)

ax[4].axis('off')

ax[5].imshow(wordcloud_tech, interpolation='bilinear')

ax[5].set_title('tech', fontsize=30)

ax[5].axis('off')
vocabulary_size = 20000

tokenizer = Tokenizer(num_words= vocabulary_size)

tokenizer.fit_on_texts(df['text'])



sequences = tokenizer.texts_to_sequences(df['text'])

data = pad_sequences(sequences, maxlen=100)

print(data.shape)
embeddings_index = dict()

f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocabulary_size, 200))

for word, index in tokenizer.word_index.items():

    if index > vocabulary_size - 1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector
df = df.replace({"category": bbc_news_categories})

bbc_news_X = data

bbc_news_Y = df['category']

X_train, X_test, Y_train, Y_test = train_test_split(bbc_news_X, bbc_news_Y, test_size=0.25)
precisions_micro = []

precisions_macro = []

recalls_micro = []

recalls_macro = []

f1s_micro = []

f1s_macro = []



def plot_confusion_matrix(Y_test, Y_pred):

    cmatrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)

    cm_fig, cm_ax = plt.subplots(figsize=(8.0, 8.0))

    cm_ax.matshow(cmatrix)



    cm_ax.set_xticklabels([''] + list(bbc_news_categories.keys()))

    cm_ax.set_yticklabels([''] + list(bbc_news_categories.keys()))



    for i in range(len(bbc_news_categories.keys())):

        for j in range(len(bbc_news_categories.keys())):

            cm_ax.text(x=j, y=i, s=cmatrix[i, j], va='center', ha='center')



    plt.title('Confusion matrix')

    plt.xlabel('Predicted categories')

    plt.ylabel('Actual categories')
def build_mlp():

    model_glove = Sequential()

    model_glove.add(Embedding(vocabulary_size, 200, input_length=100, weights=[embedding_matrix], trainable=False))

    model_glove.add(Dropout(0.3))

    model_glove.add(Conv1D(64, 5, activation='relu'))

    model_glove.add(MaxPooling1D(pool_size=4))

    model_glove.add(LSTM(units = 100))

    model_glove.add(Dense(5, activation='softmax'))

    model_glove.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_glove
model_conv = build_mlp()
history = model_conv.fit(X_train, Y_train, validation_split=0.2, epochs = 30, batch_size = 8)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
Y_pred = model_conv.predict(X_test.astype(floatx()))
Y_pred = np.argmax(Y_pred, axis = -1)
precision_micro = precision_score(y_true=Y_test, y_pred=Y_pred, average='micro')

precision_macro = precision_score(y_true=Y_test, y_pred=Y_pred, average='macro')

recall_micro = recall_score(y_true=Y_test, y_pred=Y_pred, average='micro')

recall_macro = recall_score(y_true=Y_test, y_pred=Y_pred, average='macro')

f1_micro = f1_score(y_true=Y_test, y_pred=Y_pred, average='micro')

f1_macro = f1_score(y_true=Y_test, y_pred=Y_pred, average='macro')



print('Precision score: %f (micro) / %f (macro)' % (precision_micro, precision_macro))

print('Recall score: %f (micro) / %f (macro)' % (recall_micro, recall_macro))

print('F1 score: %f (micro) / %f (macro)' % (f1_micro, f1_macro))
Y_pred.dtype
Y_test = Y_test.to_numpy(dtype ='int64')
plot_confusion_matrix(Y_test, Y_pred)