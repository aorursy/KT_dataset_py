# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.utils as ku

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import string
data_train = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")

data_val = pd.read_csv("/kaggle/input/spooky-author-identification/test.zip")



print('Training data shape:',data_train.shape)

print('Validation data shape:',data_val.shape)

data_train.head()
StopWords = set(stopwords.words('english'))



def text_preprocess(text):

    trans = str.maketrans('','',string.punctuation)

    text = text.translate(trans)

    text = ' '.join([word.lower() for word in text.split() if word.lower() not in StopWords])

    return text



data_train['text'] = data_train['text'].apply(text_preprocess)

data_val['text'] = data_val['text'].apply(text_preprocess)

data_train.head()
label_encoder = LabelEncoder()

X_train = data_train['text']

X_train = X_train.tolist()

X_test = data_val['text']

X_test = X_test.tolist()

y_train = data_train['author']

y_train = label_encoder.fit_transform(y_train)

y_train_cat = ku.to_categorical(y_train, num_classes=3)

val_id = data_val['id']



lemmatizer = WordNetLemmatizer()

X_train_lemm = []

for text in X_train:

    lem_text = ''

    for word in text.split():

        lem_word = lemmatizer.lemmatize(word, pos='v')

        lem_word = lemmatizer.lemmatize(lem_word)

        lem_text = lem_text + ' ' + lem_word

    X_train_lemm.append(lem_text)



X_test_lemm = []

for text in X_test:

    lem_text = ''

    for word in text.split():

        lem_word = lemmatizer.lemmatize(word, pos='v')

        lem_word = lemmatizer.lemmatize(lem_word)

        lem_text = lem_text + ' ' + lem_word

    X_test_lemm.append(lem_text)



tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train_lemm)

vocab_size = len(tokenizer.word_index)

max_len = 150

train_seq = tokenizer.texts_to_sequences(X_train_lemm)

train_pad = pad_sequences(train_seq, maxlen=max_len)

test_seq = tokenizer.texts_to_sequences(X_test_lemm)

test_pad = pad_sequences(test_seq, maxlen=max_len)



label2idx = {

    'EAP': 0,

    'HPL': 1,

    'MWS': 2

}
tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.5)

X_train_tfidf = tfidf.fit_transform(X_train_lemm)

X_test_tfidf = tfidf.transform(X_test_lemm)
clf = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

print(y_pred)

output_prob = clf.predict_proba(X_test_tfidf)

output_prob[:,0]
model = keras.Sequential([

    keras.layers.Embedding(vocab_size+1, 300, input_length=max_len),

    keras.layers.SpatialDropout1D(0.5),

    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),

    keras.layers.Bidirectional(keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)),

    keras.layers.Dense(300, activation='relu'),

    keras.layers.Dense(3, activation='softmax')

])

opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(train_pad, y_train_cat, epochs=20, batch_size=512)
model.summary()
y_pred_nn = model.predict_classes(test_pad)

print(y_pred_nn)

#cosine similarity between outputs from both methods.

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([y_pred], [y_pred_nn])
#Submission file.

df = pd.DataFrame()

df['id'] = val_id

df['EAP'] = output_prob[:,0]

df['HPL'] = output_prob[:,1]

df['MWS'] = output_prob[:,2]



df.to_csv('Submission.csv', index=False)