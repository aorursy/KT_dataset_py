import re
import nltk

import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
plt.style.use('ggplot')
!ls ../input
food_reviews_df=pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
food_reviews_df.shape
food_reviews_df = food_reviews_df[['Text','Score']].dropna()
ax=food_reviews_df.Score.value_counts().plot(kind='bar')
fig = ax.get_figure()
fig.savefig("score.png");
def review_to_wordlist( review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return(b)
df_5 = food_reviews_df[food_reviews_df.Score == 5].sample(n=25000)
df_4 = food_reviews_df[food_reviews_df.Score == 4].sample(n=25000)
df_3 = food_reviews_df[food_reviews_df.Score == 3].sample(n=25000)
df_2 = food_reviews_df[food_reviews_df.Score == 2].sample(n=25000)
df_1 = food_reviews_df[food_reviews_df.Score == 1].sample(n=25000)

data = pd.concat([df_1,df_2,df_3,df_4,df_5])
train, test = train_test_split(data, test_size = 0.2)
clean_train_reviews = []
for review in tqdm(train['Text']):
    clean_train_reviews.append( " ".join(review_to_wordlist(review)))
    
clean_test_reviews = []
for review in tqdm(test['Text']):
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))
max_features = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
maxlen = 80
batch_size = 32
nb_classes = 5
num_epochs = 1
# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(train['Text'])
sequences_train = tokenizer.texts_to_sequences(train['Text'])
sequences_test = tokenizer.texts_to_sequences(test['Text'])

import pickle
# saving
# with open('tokenizer-2.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

y_train = np.array(train['Score']-1)
y_test = np.array(test['Score']-1)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
from keras.callbacks import EarlyStopping, ModelCheckpoint
print('Build model...')
model_conv = Sequential()
model_conv.add(Embedding(max_features, 100, dropout=0.3))
model_conv.add(Dropout(0.3))
model_conv.add(Conv1D(64, 5, activation='relu'))
model_conv.add(MaxPooling1D(pool_size=4))
model_conv.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2))
model_conv.add(Dense(nb_classes))
model_conv.add(Activation('softmax'))

model_conv.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_json_2 = model_conv.to_json()
with open("model-cnn.json", "w") as json_file:
    json_file.write(model_json_2)
print('Model saved to Disk!')

filepath="weights-improvement-cnn-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, EarlyStopping(patience=2, monitor='val_acc')]

print('Train...')
history = model_conv.fit(X_train, Y_train,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks_list,
                 validation_split=VALIDATION_SPLIT)
## Test Time
import pickle
with open('../input/rnnlstm/tokenizer-2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

from keras.models import model_from_json
# load json and create model
json_file = open('../input/rnnlstm/model-cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../input/rnnlstm/weights-improvement-cnn-09-0.61.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

maxlen = 80
sequences_test = tokenizer.texts_to_sequences(test['Text'])
X_test_cnn = sequence.pad_sequences(sequences_test, maxlen=maxlen)

nb_classes = 5
y_test = np.array(test['Score']-1)
Y_test = np_utils.to_categorical(y_test, nb_classes)


score = loaded_model.evaluate(X_test_cnn, Y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
import pickle
with open('../input/rnnlstm/tokenizer.pickle', 'rb') as handle:
    tokenizer_rnn = pickle.load(handle)

from keras.models import model_from_json
# load json and create model
json_file = open('../input/rnnlstm/model-rnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_rnn = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_rnn.load_weights("../input/rnnlstm/weights-improvement-rnn-08-0.60.hdf5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

maxlen = 80
sequences_test = tokenizer_rnn.texts_to_sequences(test['Text'])
X_test_rnn = sequence.pad_sequences(sequences_test, maxlen=maxlen)

X_test_rnn.shape
nb_classes = 5
y_test = np.array(test['Score']-1)
Y_test = np_utils.to_categorical(y_test, nb_classes)


score = loaded_model_rnn.evaluate(X_test_rnn, Y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
pred_rnn = loaded_model_rnn.predict(X_test_rnn, verbose=1)
pred_cnn = loaded_model.predict(X_test_cnn, verbose = 1)
ensemble = (pred_cnn + pred_rnn)/2
ens = np.argmax(ensemble, axis=1)

y_test = np.array(test['Score']-1)
print(confusion_matrix(y_test, ens))
print('Classification Report')
target_names = ['1', '2', '3', '4', '5']
print(classification_report(y_test, ens, target_names=target_names))


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, ens))
def test_function(string):
    clean_string = []
    clean_string.append(" ".join(review_to_wordlist(str(string))))
    maxlen = 80
    sequences_test_rnn = tokenizer_rnn.texts_to_sequences(clean_string)
    string_rnn = sequence.pad_sequences(sequences_test_rnn, maxlen=maxlen)
    sequences_test_cnn = tokenizer.texts_to_sequences(clean_string)
    string_cnn = sequence.pad_sequences(sequences_test_cnn, maxlen=maxlen)
    pred_rnn = loaded_model_rnn.predict(string_rnn, verbose=1)
    pred_cnn = loaded_model.predict(string_cnn, verbose = 1)
    ensemble = (pred_cnn + pred_rnn)/2
    print(ensemble)

    rating = np.argmax(ensemble, axis=1) + 1
    confidence = np.max(ensemble)
    return rating, confidence
    
food_reviews_df.iloc[0,0]
s = '''
I have bought several of the Vitality canned dog food products and have found them all to 
be of good quality. The product looks more like a stew than a processed meat and it smells better. 
My Labrador is finicky and she appreciates this product better than  most.
'''
r, c = test_function(s)
print( r, c)
