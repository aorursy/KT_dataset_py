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
import tensorflow as tf

from numpy import array

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Embedding

docs = ['너무 재밌네요', '최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다.', '한번 더 보고싶네요.', '글쎄요', '별로예요', '생각보다 지루하네요', '연기가 어색해요', '재미없어요']
classes = array([1,1,1,1,1,0,0,0,0,0])
token = Tokenizer()

token.fit_on_texts(docs)

print(token.word_index)
x = token.texts_to_sequences(docs)

print(x)
padded_x = pad_sequences(x, 4)

"\n패딩 결과\n", print(padded_x)
word_size = len(token.word_index)+1
model = Sequential()

model.add(Embedding(word_size, 8, input_length=4))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padded_x, classes, epochs=20)



print("\n Accuracy: %.4f" % (model.evaluate(padded_x, classes)[1]))