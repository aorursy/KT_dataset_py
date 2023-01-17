import numpy as np 

import pandas as pd

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(train.text.values)

word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(train.text.values)

X = pad_sequences(X, maxlen=250)

Y = train.target
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Conv1D, MaxPooling1D



model = Sequential()

model.add(Embedding(50000, 128, input_length=X.shape[1]))

model.add(Dropout(0.2))

model.add(Conv1D(filters=64,kernel_size=5, activation='relu', padding="valid", strides = 1))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])



epochs = 4

batch_size = 64



history = model.fit(X, Y, epochs=epochs, batch_size=batch_size,validation_split=0.1)
X_submit = tokenizer.texts_to_sequences(test.text.values)

X_submit = pad_sequences(X_submit, maxlen=250)

y_submit = model.predict_classes(X_submit)
submission_df = pd.DataFrame({'id':test.id, 'target':y_submit[:,0]})

submission_df.to_csv('submission.csv', index = False)