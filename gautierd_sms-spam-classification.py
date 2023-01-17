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
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import confusion_matrix

import seaborn as sns

from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics import accuracy_score

from sklearn.decomposition import TruncatedSVD

from IPython.display import FileLink
dataDir = '/kaggle/input/codemlnaivebayes/'
train = pd.read_csv(dataDir+'train.csv', encoding='latin-1')

test = pd.read_csv(dataDir+'test_cleaned.csv',encoding='latin-1')

sample_submission = pd.read_csv(dataDir+'sample_submission.csv')
train.head()
targets = train.v1.values
print('{} % Only of spam of {} texts '.format( np.sum(targets=='spam')/len(targets)*100,len(targets)))
test.head()
ham_spam_index = {'ham':0,'spam':1}
stemmer = SnowballStemmer("english")

train_message_stem = [ ' '.join([stemmer.stem(word) for word in  el.split(' ')]) for el in train.v2]

test_message_stem = [ ' '.join([stemmer.stem(word) for word in  el.split(' ')]) for el in test.message]
X = train_message_stem

X_test = test_message_stem

y = train.v1.apply(lambda el: ham_spam_index[el]).values
train_message_stem[:5]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
params = {'ngram_range':(1,4),'analyzer':'char_wb'} # Character level language model 
vectorizer = CountVectorizer(**params)

X_train_bow = vectorizer.fit_transform(X_train).todense()

X_val_bow = vectorizer.transform(X_val).todense()
X_train_bow.shape
gnb = GaussianNB()

y_pred_bow = gnb.fit(X_train_bow, y_train).predict(X_val_bow)

acc= accuracy_score(y_val,y_pred_bow)

print(acc)
sns.heatmap(confusion_matrix(y_val, y_pred_bow),annot=True,fmt='d')
from keras.layers import Input, Dense, Dropout, GaussianNoise

from keras.models import Model

import tensorflow as tf



# this is the size of our encoded representations

encoding_dim = 16  # Very small compared to the bag of words representation



# this is our input placeholder

input_sms = Input(shape=(X_train_bow.shape[1],))



#Introduce noise to make the model generalise

noise_input_sms = GaussianNoise(stddev=0.3)(input_sms) 



# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(noise_input_sms)



# "decoded" is the lossy reconstruction of the input

decoded = Dense(X_train_bow.shape[1], activation='linear',name='reconstruction')(encoded)



encoded = Dropout(0.1)(encoded)



sentiment_proj = Dense(encoding_dim,activation='tanh')(encoded)

sentiment_proj = Dropout(0.1)(encoded)



sentiment_out = Dense(2,activation='softmax',name='spam')(sentiment_proj)



# this model maps an input to its reconstruction

autoencoder = Model(input_sms, decoded)



# this model maps an input to its encoded representation

encoder = Model(input_sms, encoded)



autoencoder.compile(optimizer='adam', loss='mse')



full_model = Model(inputs=[input_sms],outputs=[sentiment_out,decoded])

full_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 0.1])
full_model.fit([X_train_bow], [tf.one_hot(y_train,depth=2), X_train_bow],

                epochs=6,

                steps_per_epoch=20,

                validation_steps=2,

                shuffle=True,

                validation_data=([X_val_bow], [tf.one_hot(y_val,depth=2), X_val_bow]))
y_pred_val_logits = full_model.predict(X_val_bow)

y_pred_neural = [np.argmax(el) for el in y_pred_val_logits[0]]

acc=accuracy_score(y_val,y_pred_neural)

print(acc)
X_test_submission = vectorizer.transform(X_test).todense()

X_test_submission = vectorizer.transform(X_test).todense()

y_test_val_logits = full_model.predict(X_test_submission)

y_test_neural = [np.argmax(el) for el in y_test_val_logits[0]]

submission = pd.DataFrame({'Id':test.Id,'Spam':y_test_neural})

submission.to_csv('submission.csv', index=False)
sns.heatmap(confusion_matrix(y_val, y_pred_neural),annot=True,fmt='d')