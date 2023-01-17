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
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



stop_words = set(stopwords.words('english'))

snowball_stemmer = SnowballStemmer("english")
train_df = pd.read_csv("../input/train.csv")

#test_df = pd.read_csv("../input/test.csv")

train_df.columns
train_df.head(10)
train_df['target'] = [ 1 if target>=0.5 else 0 for target in train_df['target'] ]

#train_df.head(10)
train_df['target'].value_counts()
x_label = ('Toxic comments', 'Non-toxic comments')

y_axis =[ value_count/train_df.shape[0]*100 for value_count in train_df['target'].value_counts().tolist() ]

#y_label = np.array(y_axis)

bar_plot = plt.bar(x_label, y_axis)

bar_plot[0].set_color('r')

bar_plot[1].set_color('g')

plt.ylabel('Total data range', fontsize=10)
X = train_df['comment_text']

y = train_df['target']
import re

def pre_processing(X):

    X = X.str.lower()

    X = X.str.replace(r'\r', ' ')

    X = X.str.replace(r'\n', ' ')

    X = X.str.replace('[^a-zA-Z0-9 ]', '')

    #X = re.sub('[^a-zA-Z0-9 \n\.]','', X)

    return X
X = pre_processing(X)
def nlp_preprocessing(document): 

    words = [snowball_stemmer.stem(word) for word in document.split() 

                 if word not in stop_words]

    doc = ' '.join(words)

    return doc
x = X.apply(lambda document: nlp_preprocessing(document))
vectorizer = TfidfVectorizer(min_df=0.01)

x = vectorizer.fit_transform(x)
lencoder= LabelEncoder()

enc_y = lencoder.fit_transform(y)
import tensorflow as tf

import keras.backend as K

from keras import layers, models, optimizers, regularizers

from keras.layers import Bidirectional, LSTM

from keras.models import Sequential
nlp_input = layers.Input((x.shape[1], ), sparse=True)

hidden_layer_1 = layers.Dense(500, activation="relu")(nlp_input)



hidden_drop_1 = layers.Dropout(0.3)(hidden_layer_1)

output_layer = layers.Dense(1, activation="sigmoid")(hidden_drop_1)



classifier_1 = models.Model(inputs = nlp_input, outputs = output_layer)

classifier_1.compile(optimizer=optimizers.adam(lr=0.001, amsgrad=True), 

                   loss='binary_crossentropy', 

                   metrics=['accuracy'])
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)



                       
classifier_1.fit(x, enc_y, batch_size = 256, epochs = 10, callbacks=[es, mc], verbose=1  )
test_df = pd.read_csv("../input/test.csv")
test_X = pre_processing(test_df['comment_text'])

test_x = test_X.apply(lambda document: nlp_preprocessing(document))

test_x = vectorizer.transform(test_x)
y = classifier_1.predict(test_x)

y = np.where(y>=0.5,1,0)

sub_df = pd.read_csv('../input/sample_submission.csv')

sub_df['prediction'] = y
sub_df.to_csv('submission.csv',index = False)