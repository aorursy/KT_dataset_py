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
import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.layers import Embedding,Bidirectional

from sklearn import preprocessing 

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.preprocessing import sequence, text

import os

from nltk.corpus import stopwords

import re

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

train.drop(['id', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

train.head()
stop_words = set(stopwords.words('english'))



def data_text_preprocess(total_text, ind, col):

    # Remove int values from text data as that might not be imp

    if type(total_text) is not int:

        string = ""

        # replacing all special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replacing multiple spaces with single space

        total_text = re.sub('\s+',' ', str(total_text))

        # bring whole text to same lower-case scale.

        total_text = total_text.lower()

        

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from text

            if not word in stop_words:

                string += word + " "

        

        train[col][ind] = string

        

for index, row in train.iterrows():

    if type(row['comment_text']) is str:

        data_text_preprocess(row['comment_text'], index, 'comment_text')
train.head()
xtrain, xvalid, ytrain, yvalid = train_test_split( train.comment_text.values, train.toxic.values, 

                                                   stratify=train.toxic.values, 

                                                   random_state=42, 

                                                   test_size=0.2, shuffle=True)
token = text.Tokenizer(num_words=None)

token.fit_on_texts(list(xtrain) + list(xvalid))

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)
max_len = 100

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
xtrain_pad[200]
word_index = token.word_index
model = Sequential()

model.add(Embedding(len(word_index) + 1,256,input_length=max_len))

model.add(Bidirectional(LSTM(64)))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

    

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   
%time model.fit(xtrain_pad, ytrain, epochs=2, batch_size = 128, validation_data=(xvalid_pad, yvalid))
scores = model.predict(xvalid_pad)



test_loss, test_acc = model.evaluate(scores,  yvalid, verbose=1)

print('\nTest accuracy:', test_acc)
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test.head()
def data_text_preprocess(total_text, ind, col):

    # Remove int values from text data as that might not be imp

    if type(total_text) is not int:

        string = ""

        # replacing all special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replacing multiple spaces with single space

        total_text = re.sub('\s+',' ', str(total_text))

        # bring whole text to same lower-case scale.

        total_text = total_text.lower()

        

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from text

            if not word in stop_words:

                string += word + " "

        

        test[col][ind] = string

        

for index, row in test.iterrows():

    if type(row['content']) is str:

        data_text_preprocess(row['content'], index, 'content')
test.head()
test_data = token.texts_to_sequences(test.content)

test_data_seq = sequence.pad_sequences(test_data, maxlen=max_len)
test['toxic'] = model.predict(test_data_seq, verbose=1)
test[['id', 'toxic']].to_csv('submission.csv', index=False)