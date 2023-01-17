# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

vocab_size = 500

df['encoded_text'] = [one_hot(x, vocab_size) for x in df['text']]
# pad documents to a max length of 4 words

max_length = 256

padded_text = pad_sequences(df['encoded_text'], maxlen=max_length, padding='post')
padded_text[0]
from keras import regularizers, optimizers

from keras.layers import BatchNormalization

# define the model

model = Sequential()

model.add(Embedding(vocab_size, 50, input_length=max_length))

model.add(Flatten())

model.add(BatchNormalization())

#model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

model.add(Dense(400, activation='elu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# compile the model

sgd = optimizers.SGD(lr=0.018, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())
import keras

from keras.layers import Dense, GlobalAveragePooling1D, Embedding

import keras.backend as K

from keras.callbacks import EarlyStopping

from keras.models import Sequential



from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

a2c = {'EAP':0, 'HPL':1, 'MWS':2}

labels = to_categorical([a2c[x] for x in df['author']])
labels[:10]
# fit the model

model.fit(padded_text, labels, epochs=50, verbose=1,batch_size=256, validation_split=0.33)
df_test = pd.read_csv('../input/test.csv')

df_test['encoded_text'] = [one_hot(x, vocab_size) for x in df_test['text']]

padded_text_test = pad_sequences(df_test['encoded_text'], maxlen=max_length, padding='post')

y_pred = model.predict_proba(padded_text_test)



result = pd.read_csv('../input/sample_submission.csv')

for a, i in a2c.items():

    result[a] = y_pred[:, i]
result.to_csv('out.csv')
print(one_hot('man', 2))

print(one_hot('woman', 2))
