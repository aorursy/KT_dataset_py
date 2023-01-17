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

df = pd.read_csv("/kaggle/input/imdb-reviews/dataset.csv",encoding ='latin1')

df.head(10)
# Contagem do número de observações do dataset

df.count()
# Agrupamento pelo atributo "Sentiment"

print(df.Sentiment.sum())
# Resumo

df.describe()
import re



#cleaned_chars_df = df["Sentiment"].copy()

X_cleaned = (df['SentimentText'].apply(lambda x : re.sub("\s", " ", re.sub("[^a-zA-z,\.\s]", "", x)).lower()))

X_cleaned.head()



#x = np.array(cleaned_chars_df)

#print(x)
# Model Parameters



BATCH_SIZE = 128

EPOCHS = 5

CHARSET_SIZE = 26+3





Y_cleaned = df.iloc[:, 1:]

print(X_cleaned, Y_cleaned)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=CHARSET_SIZE, lower=True,split=' ')



tokenizer.fit_on_texts(X_cleaned)

sequences = tokenizer.texts_to_sequences(X_cleaned)



padded_sequences = pad_sequences(sequences, maxlen=500)


from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout

from sklearn.model_selection import train_test_split

import re

import numpy as np 

import pandas as pd

from nltk.corpus import stopwords

from nltk import word_tokenize

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, Y_cleaned, stratify=Y_cleaned['Sentiment'], test_size=0.2)

print(X_train, Y_train)
n_max_palavras = 5000

embedding_dimensions = 100

model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=padded_sequences.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100))

model.add(Dropout(0.2))

model.add(Dense(1, activation="relu"))



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())
epochs = 5

batch_size = 512



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
scores = model.evaluate(X_test, Y_test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))