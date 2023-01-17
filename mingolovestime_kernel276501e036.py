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



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer



from keras.models import Sequential

from keras.layers import Dense

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras import layers

from keras.layers import Dense, LSTM, Embedding

# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))

train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

# test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

# submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")
print('features')

    

comments = 'comment_text'

train[comments].fillna("NULL", inplace=True)

# test[comments].fillna("NULL", inplace=True)



tokenizer = Tokenizer(num_words=80000, 

                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 

                                   lower=True, 

                                   split=' ', 

                                   char_level=False, 

                                   oov_token=None, 

                                   document_count=0)



tokenizer.fit_on_texts(train[comments])

# tokenizer.fit_on_texts(test[comments])



print('start padded')

MAX_LENGTH = 1000

padded_train_sequences = pad_sequences(tokenizer.texts_to_sequences(train[comments]), maxlen=MAX_LENGTH)

# padded_test_sequences = pad_sequences(tokenizer.texts_to_sequences(test[comments]), maxlen=MAX_LENGTH)

y = np.where(train['target'] >= 0.5, 1, 0)



print('x&y featuring finished')



model = Sequential()

model.add(Embedding(input_dim = 80000, output_dim = 300))

model.add(layers.LSTM(256))

model.add(layers.Dense(units = 64))

model.add(layers.Dense(units = 32, activation = "tanh"))

model.add(layers.Dense(units = 32))

model.add(layers.Dense(units = 8, activation = "tanh"))

model.add(layers.Dense(units = 8))

model.add(layers.Dense(units = 8, activation = "sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
print('start fit')

model.fit(padded_train_sequences[:-500000], y[:-500000], 

          epochs=2, batch_size=512, verbose=1, 

          validation_data=(padded_train_sequences[-500000:], y[-500000:]))
print('start test')

test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

tokenizer.fit_on_texts(test[comments])

padded_test_sequences = pad_sequences(tokenizer.texts_to_sequences(test[comments]), maxlen=MAX_LENGTH)
y_pred_rnn_simple = model.predict(padded_test_sequences, verbose=1, batch_size=128)
submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")
y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=['prediction'])



print('submission')

submid = pd.DataFrame({'id': submission["id"]})

submid['prediction'] = y_pred_rnn_simple['prediction']

submid.to_csv('submission.csv', index=False)