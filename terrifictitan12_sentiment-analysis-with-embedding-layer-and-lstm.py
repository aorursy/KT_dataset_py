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
data = pd.read_csv("/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv")
data.head()
import string
def preprocess_data(text):
    text = text.lower()
    
    text = [w for w in text if w not in string.punctuation]
    
    return "".join(text)
data['Text'] = data['Text'].apply(preprocess_data)
data
len(max(data['Text']))
len(min(data['Text']))
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['Text'], data['Sentiment'], test_size=0.2, random_state=42)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train_ = tokenizer.texts_to_sequences([w for w in x_train])
x_test_ = tokenizer.texts_to_sequences([w for w in x_test])
x_test_
MAXLEN = 70
x_train_ = pad_sequences(x_train_, maxlen=MAXLEN)
x_test_ = pad_sequences(x_test_, maxlen=MAXLEN)
x_train_
import keras
from keras import layers
len(set(x_train))
len(set(x_test))
len(set(y_train))
model = keras.models.Sequential()
model.add(layers.Embedding(11000, 128, input_length=MAXLEN))
model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train_, y_train, epochs=40, batch_size=128)
model.evaluate(x_test_, y_test)
pred = model.predict(x_test_)
pred
example = 'Why the hell stocks are not increasing, i am so angry right now !!!!!'
example = example.lower()
example = [w for w in example if w not in string.punctuation]
example = "".join(example)
example
toke = Tokenizer()
toke.fit_on_texts(example)
example_ = toke.texts_to_sequences(example)
example_ = pad_sequences(example_, MAXLEN)
example_
len(example_)
model.predict_classes(example_)
