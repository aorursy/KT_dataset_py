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
import os



# Time

import time

import datetime



# Numerical

import numpy as np

import pandas as pd



# Tools

import itertools

from collections import Counter



# NLP

import re

from nltk.corpus import stopwords



# Preprocessing

from sklearn import preprocessing

from sklearn.utils import class_weight as cw

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# Model Selection

from sklearn.model_selection import train_test_split



# Evaluation Metrics

from sklearn import metrics 

from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report



# Deep Learing Preprocessing - Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical



# Deep Learning Model - Keras

from keras.models import Model

from keras.models import Sequential



from keras.layers import Dense, Embedding

from keras.models import Sequential



# Deep Learning Model - Keras - RNN

from keras.layers import Embedding, LSTM, Bidirectional



# Deep Learning Model - Keras - General

from keras.layers import Input, Add, concatenate, Dense, Activation, BatchNormalization, Dropout, Flatten

from keras.layers import LeakyReLU, PReLU, Lambda, Multiply



from keras.preprocessing import sequence



# Deep Learning Parameters - Keras

from keras.optimizers import RMSprop, Adam



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/Tweets.csv')

df.head()
df = df[['text','airline_sentiment']]

df.head()
df = df.reindex(np.random.permutation(df.index))

df.head()
import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



def text_cleaning(tweet):

    letters = re.sub("^a-zA-Z"," ",tweet)

    ht = re.sub(r'http\S+', '',letters)

    mention = re.sub(r'@\w+', '', ht)

    p = re.sub(r'[^\w\s]','',mention)

    words = p.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return( " ".join(meaningful_words))

df['text_clean']=df['text'].apply(lambda x: text_cleaning(x))
Y = df['airline_sentiment']

lenc = LabelEncoder()

Y = lenc.fit_transform(Y)

Y = to_categorical(Y)
X_train, X_test, Y_train, Y_test = train_test_split(df['text_clean'], Y, test_size=0.2,random_state=37)
max_words = len(set(" ".join(X_train).split()))

max_len = X_train.apply(lambda x: len(x)).max()

max_words, max_len
tk = Tokenizer(num_words=max_words)

tk.fit_on_texts(X_train)

X_train_tk = tk.texts_to_sequences(X_train)

X_test_tk = tk.texts_to_sequences(X_test)
X_train_pad = sequence.pad_sequences(X_train_tk, maxlen=max_len)

X_test_pad = sequence.pad_sequences(X_test_tk, maxlen = max_len)
num_classes = 3
def get_weight(y):

    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)

    return class_weight_current

class_weight = get_weight(Y_train.flatten())
base_model = Sequential()

base_model.add(Embedding(max_words, 8, input_length=max_len))

base_model.add(Flatten())

base_model.add(Dense(64, activation='relu'))

base_model.add(Dense(64, activation='relu'))

base_model.add(Dense(3, activation='softmax'))

base_model.summary()
def deep_lr_model(model):

    batch_size = 512

    epochs = 20





    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    

    history = model.fit(X_train_pad, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1,class_weight=class_weight)

    

    return history

base_history = deep_lr_model(base_model)
score = []

score_base = base_model.evaluate(X_test_pad, Y_test)

score.append(score_base[1]*100)

score_base[1]*100

ypred1 = base_model.predict_classes(X_test_pad,verbose=1)

ypred1
reg_model = Sequential()

reg_model.add(Embedding(max_words, 8, input_length=max_len))

reg_model.add(Flatten())

reg_model.add(Dense(64, activation='relu'))

reg_model.add(Dense(3, activation='softmax'))

reg_model.summary()
reg_history = deep_lr_model(reg_model)
score_reg = reg_model.evaluate(X_test_pad, Y_test)

score.append(score_reg[1]*100)

score_reg[1]*100

drop_model = Sequential()

drop_model.add(Embedding(max_words, 8, input_length=max_len))

drop_model.add(Flatten())

drop_model.add(Dense(64, activation='relu'))

drop_model.add(Dropout(0.5))

drop_model.add(Dense(3, activation='softmax'))

drop_model.summary()
dropout_history = deep_lr_model(drop_model)
score_drop = drop_model.evaluate(X_test_pad, Y_test)

score.append(score_drop[1]*100)

score_drop[1]*100
from keras import regularizers



drop1_model = Sequential()

drop1_model.add(Embedding(max_words, 8, input_length=max_len))

drop1_model.add(Flatten())

drop1_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

drop1_model.add(Dense(3, activation='softmax'))

drop1_model.summary()
rgz_history = deep_lr_model(drop1_model)
score_drop1 = drop1_model.evaluate(X_test_pad, Y_test)

score.append(score_drop1[1]*100)

score_drop1[1]*100
def plot_performance(history, metric_name):

    metric = history.history[metric_name]

    val_metric = history.history['val_'+metric_name]

    ep = range(1,21)

    plt.plot(ep,metric,label="Training Accuracy")

    plt.plot(ep,val_metric,label="Validation Accuracy")

    plt.legend()

    plt.show()
plot_performance(base_history,'acc')

plot_performance(base_history,'loss')
plot_performance(dropout_history,'acc')

plot_performance(dropout_history,'loss')
plot_performance(reg_history,'acc')

plot_performance(reg_history,'loss')
plot_performance(rgz_history,'acc')

plot_performance(rgz_history,'loss')
def test_model(model, epoch_stop):

    model.fit(X_train_pad

              , Y_train

              , epochs=epoch_stop

              , batch_size=512

              , verbose=1)

    results = model.evaluate(X_test_pad, Y_test)

    

    return results

base_results = test_model(base_model, 10)

print('/n')

print('Test accuracy of baseline model: {0:.2f}%'.format(base_results[1]*100))
drop_results = test_model(drop_model, 10)

print('/n')

print('Test accuracy of baseline model: {0:.2f}%'.format(drop_results[1]*100))
drop1_results = test_model(drop1_model, 10)

print('/n')

print('Test accuracy of baseline model: {0:.2f}%'.format(drop1_results[1]*100))
reg_results = test_model(reg_model, 10)

print('/n')

print('Test accuracy of baseline model: {0:.2f}%'.format(reg_results[1]*100))
l = ['base model','reduced model', 'dropout model','regularized model']

index = [1,2,3,4]

plt.bar(index,score,color='rgcy')

plt.xticks(index,['base model','reduced model', 'dropout model','regularized model'],rotation=90)

plt.show()
