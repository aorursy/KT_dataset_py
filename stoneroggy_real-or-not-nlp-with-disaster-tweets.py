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
#loading datasets
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
print('shape of train and test sets are:',train.shape, test.shape)
print('features of train set:',train.columns)
print("features of test set:", test.columns)

# we can clearly see that we have to predict target value
# importing re (regex)
import re

# removing URL's from text

def remove_urls(text):
    return re.sub(r"http\S+", "", text)
    """ re.sub( a, b , c)
            -> it means that, in text 'c' we find 'a' and replace it with 'b'
    """
        

train['text'] = train['text'].apply(remove_urls)
test['text'] = test['text'].apply(remove_urls)

# removing stopwords (words like "I","we","have" etc.)
# also removing punctuations

! pip install nlppreprocess
from nlppreprocess import NLP

nlp = NLP()

train['text'] = train['text'].apply(nlp.process)
test['text'] = test['text'].apply(nlp.process)


# tokenization of text
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(train['text'])
sequences = tokenizer.texts_to_sequences(train['text'])

tokenizer_test = Tokenizer(num_words = 5000)
tokenizer_test.fit_on_texts(test['text'])
sequences_test = tokenizer_test.texts_to_sequences(test['text'])

# since all the sequences are of different length so we need to pad the sequence if small and trim if its big
#we can use keras.preprocessing.sequence pad_sequences

from keras.preprocessing.sequence import pad_sequences
input_tensor = pad_sequences(sequences,maxlen = 30)
test_tensor = pad_sequences(sequences_test, maxlen = 30)

#train test split of data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_tensor, train['target'], test_size = 0.3)
# building a model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 64))
model.add(keras.layers.LSTM(64,recurrent_dropout = 0.5, return_sequences = True))
model.add(keras.layers.LSTM(64,return_sequences = True))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(1,activation = 'sigmoid'))
model.summary()
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
model.fit(x_train, y_train, batch_size = 100, epochs = 15)
evaluation = model.evaluate(x_test, y_test)
evaluation
ids = np.array(test.id)
ids
ans = model.predict_classes(test_tensor)
ans = ans.flatten()
df = pd.DataFrame({'id':ids, 'target':ans})
df.to_csv('answer.csv')