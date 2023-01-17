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
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
list_of_clases=[0,1,2,3,4]
max_f=10000
max_t_l=400
embedding_dims=50
filters=250
kernel_size=3
hidden_dims=250
batch_size=32
epochs=7
train_df=pd.read_csv("../input/train.tsv", sep="\t")
print(train_df.head())

test_df = pd.read_csv('../input/test.tsv',  sep="\t")
print(test_df.head())
print(train_df.iloc[0,2])
print(np.where(pd.isnull(train_df)))
x= train_df["Phrase"].values
print(x)

print("properties of x")
print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(x), x.ndim, x.shape, x.size, x.dtype, x.itemsize))
y = train_df["Sentiment"].values
print(y)
print("properties of y")
print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(y), y.ndim, y.shape, y.size, y.dtype, y.itemsize))
x_tokenizer=text.Tokenizer(num_words=max_f)
print(x_tokenizer)
x_tokenizer.fit_on_texts(list(x))
x_tokenized=x_tokenizer.texts_to_sequences(x)
x_train_val=sequence.pad_sequences(x_tokenized, maxlen=max_t_l)
print("properties of x_train_val")
print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(x_train_val), x_train_val.ndim, x_train_val.shape, x_train_val.size, x_train_val.dtype, x_train_val.itemsize))
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.1, random_state=1)

model= Sequential()
model.add(Embedding(max_f,
                   embedding_dims,
                   input_length=max_t_l))
model.add(Dropout(0.2))

model.add(Conv1D(32,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_val, y_val))
x_test = test_df['Phrase'].values
print(x_test)
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_t_l)

y_testing = model.predict(x_testing, verbose = 1)

sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")

predictions = np.round(np.argmax(y_testing, axis=1)).astype(int)
# for blending if necessary.
#(ovr.predict(test_vectorized) + svc.predict(test_vectorized) + np.round(np.argmax(pred, axis=1)).astype(int)) / 3
sub['Sentiment'] = predictions
sub.to_csv("movie.csv", index=False)
