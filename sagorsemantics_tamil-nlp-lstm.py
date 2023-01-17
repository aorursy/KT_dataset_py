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

import matplotlib.pyplot as plt

import nltk

import seaborn as sns

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import SpatialDropout1D

from keras.utils import to_categorical

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from sklearn.feature_selection import RFE

import re

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('/kaggle/input/tamil-nlp/tamil_news_train.csv')

test = pd.read_csv('/kaggle/input/tamil-nlp/tamil_news_test.csv')
# fix random seed for reproducibility

np.random.seed(7)

train = train.drop_duplicates().reset_index(drop=True)

test = test.drop_duplicates().reset_index(drop=True)


train.NewsInTamil = train.NewsInTamil.str.replace('\d+', ' ')



test.NewsInTamil = test.NewsInTamil.str.replace('\d+', ' ')



    

train = train.append(test)

df = train

df.head()

df.shape
df.Category.unique()
df.Category = df.Category.replace('world', 1)

df.Category = df.Category.replace('cinema', 2)

df.Category = df.Category.replace('tamilnadu', 3)

df.Category = df.Category.replace('india', 4)

df.Category = df.Category.replace('politics', 5)

df.Category = df.Category.replace('sports', 6)





df.Category.head()
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 32000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 120

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=False)

tokenizer.fit_on_texts(df.NewsInTamil.values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(df.NewsInTamil.values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(df.Category).values

print('Shape of label tensor:', Y.shape)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=.10)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_features, train_labels, epochs=5, batch_size=32,validation_split=0.2)

# Final evaluation of the model

model_pred_train = model.predict(train_features)

model_pred_test = model.predict(test_features)

# print(classification_report(test_labels,model_pred_test))

print('LSTM Recurrent Neural Network baseline: ' + str(roc_auc_score(train_labels, model_pred_train)))

print('LSTM Recurrent Neural Network: ' + str(roc_auc_score(test_labels, model_pred_test)))
model.save_weights('tamil_news_classification.h5')
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
news = ['இயற்கையை நேசிப்பதுதானே கொண்டாட்டம்.. இது ஒரு புது முயற்சி..!']

seq = tokenizer.texts_to_sequences(news)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(padded)

labels = ['world', 'cinema', 'tamilnadu', 'india', 'politics', 'sports']

label = pred, labels[np.argmax(pred)]

print("News Category is: ")

print(label[1])