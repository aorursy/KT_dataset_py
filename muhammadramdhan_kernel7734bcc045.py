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
%ls /kaggle/input/training-nationality-name
from keras.models import Sequential
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Flatten, Activation, Dropout, MaxPool1D, Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import pickle
import string

input_length = 42

data = pd.read_csv("/kaggle/input/training-nationality-name/train.csv", sep=",",  dtype={"nama":str,"country":int})
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['nama'])
pickle.dump(tokenizer, open("tokenizer.pc", "wb+")) # save tokenizer object
X = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), input_length, padding='post')
#X = np.expand_dims(X, axis=2)
Y = data['country'].values

filters = 148
kernel_size = 4
model = Sequential()

model.add(Embedding(len(tokenizer.word_index) + 1, 48, input_length=input_length))
model.add(Conv1D(192, 5, activation='relu', strides=1))
model.add(Conv1D(384, 4, activation='relu', strides=1))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.6))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dropout(0.6))
model.add(Activation('relu'))
model.add(Dense(18, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=130, batch_size=200)
data = pd.read_csv("/kaggle/input/training-nationality-name/evaluation.csv", sep=",",  dtype={"nama":str,"country":int})
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), input_length, padding='post')
#X_test = np.expand_dims(X_test, axis=2)
Y_test = data['country'].values
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
!md5sum model.h5
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")