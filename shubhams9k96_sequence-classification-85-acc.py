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

pd.set_option('display.max_colwidth', -1)



# Any results you write to the current directory are saved as output.
from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Conv1D, Dropout

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

df.info()
df.head()
df_train, df_test = train_test_split(df, test_size=0.3)

print(df_train.shape)

print(df_test.shape)
df_train['sentiment'] = df_train['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

df_test['sentiment'] = df_test['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
df_train['review'] = df_train['review'].str.replace('<br />', '')

df_test['review'] = df_test['review'].str.replace('<br />', '')

df_train.head()
y_train = df_train['sentiment']

X_train = df_train['review']

y_test = df_test['sentiment']

X_test = df_test['review']

max_review_length = 500
# X_train = sequence.pad_sequences(X_train.values, maxlen = 500)

# X_test = sequence.pad_sequences(X_test)
from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_train = sequence.pad_sequences(X_train, maxlen=1000)
X_test = tokenizer.texts_to_sequences(X_test)

X_test = sequence.pad_sequences(X_test, maxlen=1000)
print(X_train.shape)

print(X_test.shape)
model = Sequential()

model.add(Embedding(1000, 64, input_length=X_train.shape[1]))

model.add(LSTM(200))

model.add(Dropout(0.25))

# model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history= model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_data=(X_test, y_test))
def plot_history(history):

    fig = plt.figure(figsize = (20, 6))

    plt.subplot(1, 2, 1)

    plt.plot(history.history['acc'], label='Train Acc')

    plt.plot(history.history['val_acc'], label='Validation Acc')

    plt.title("Accuracy")

    plt.legend()

    plt.grid()

    

    plt.subplot(1, 2, 2)

    plt.plot(history.history['loss'], label='Train loss')

    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.title("Loss")

    plt.legend()

    plt.grid()

    

    plt.show()
plot_history(history)
model.save('model-lstm.h5')
!ls
import pickle 
with open( 'tokenizer.pkl', 'wb') as f:

    pickle.dump(tokenizer, f)