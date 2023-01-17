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
#reading the file

import numpy as np

import pandas as pd

import spacy

from spacy import displacy



loc = '/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'

df = pd.read_csv(loc)



nlp = spacy.load('en')



# 1 for positve Sentiment

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['sentiment'] = le.fit_transform(df['sentiment'])



df['review'] = df['review'].apply(lambda x:x.replace('<br /><br />', ''))

df['review'] = df['review'].apply(lambda x:x.replace('..', ''))

df['review'] = df['review'].apply(lambda x:x.replace('...', ''))

df['review'] = df['review'].apply(lambda x:x.replace('....', ''))
import string



punct = string.punctuation



def tokenizer_f(x):

    sent = nlp(x)

    token = []

    for i in (sent):

        if i.lemma_ == '-PRON-':

            token.append(i.lower_)

        elif not i.is_stop and not i.lemma_.lower() in punct:

            token.append(i.lemma_.lower())          

    return ' '.join(token)

            
from tqdm import tqdm

review = []

for i in tqdm(range(df.shape[0])):

    review.append(tokenizer_f(df['review'][i]))
X = np.array(review)

y = df['sentiment'].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify = y)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=10000,oov_token="<00v>")



tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)



X_train = pad_sequences(X_train, maxlen=400)

X_test = pad_sequences(X_test, maxlen=400)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalAvgPool1D
model = Sequential()

model.add(Embedding(input_dim = 10000, output_dim = 15,input_length = 400))

model.add(GlobalAvgPool1D())

model.add(Dense(15, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])



model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)