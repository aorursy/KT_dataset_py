import pandas as pd

import numpy as np

np.random.seed(42)
df = pd.read_csv(filepath_or_buffer='../input/roman_numerals.txt', dtype='str')
df.head()
df = df.sample(frac=1, random_state=42)
roman_values = df.loc[:, 'ROMAN'].astype(np.str)

ha_values = df.loc[:, 'HA'].astype(np.str)
import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
roman_tokenizer = Tokenizer(char_level=True)

roman_tokenizer.fit_on_texts(roman_values)



ha_tokenizer = Tokenizer(char_level=True)

ha_tokenizer.fit_on_texts(ha_values)
roman_tokenizer.word_index, ha_tokenizer.word_index
roman_index_to_char = {v:k for k,v in roman_tokenizer.word_index.items()}

roman_index_to_char
ha_index_to_char = {v:k for k,v in ha_tokenizer.word_index.items()}

ha_index_to_char
# Non Spatial Inputs

x_roman_1 = roman_tokenizer.texts_to_matrix(roman_values)

x_roman_2 = roman_tokenizer.texts_to_matrix(roman_values, mode='count')

# Spatial Inputs

x_roman_3 = pad_sequences(roman_tokenizer.texts_to_sequences(roman_values), padding='pre')
x_roman_3.shape
num_ex = x_roman_3.shape[0]

roman_seq_len = x_roman_3.shape[1]

roman_vocab_size = len(roman_index_to_char) + 1
arr = [to_categorical(x_roman_3[i], num_classes=roman_vocab_size) for i in range(num_ex)]

arr = [item.reshape(roman_seq_len * roman_vocab_size,) for item in arr]

temp = np.vstack(arr)

x_roman_3 = temp.reshape((num_ex,roman_seq_len,roman_vocab_size))
# Shape of Spatial Inputs

x_roman_3.shape
# Shape of Non Spatial Inputs

x_roman = np.hstack([x_roman_1, x_roman_2])
x_roman.shape
y_ha = ha_tokenizer.texts_to_sequences(ha_values)

y_ha = pad_sequences(y_ha, padding='post')
ha_seq_len = y_ha.shape[1]

ha_vocab_size = len(ha_index_to_char) + 1
arr = [to_categorical(y_ha[i], num_classes=ha_vocab_size) for i in range(num_ex)]

arr = [item.reshape(ha_seq_len * ha_vocab_size,) for item in arr]

temp = np.vstack(arr)

y_ha = temp.reshape((num_ex,ha_seq_len,ha_vocab_size))
# Shape of Output Labels

y_ha.shape
from keras.models import Sequential, Model

from keras.layers import Dense, Activation, SimpleRNN, LSTM, RepeatVector, Flatten,  Input, Reshape
NON_SPATIAL_INPUT_DIM = x_roman.shape[1]

# OUTPUT_DIM = y_ha.shape[1]
input1_nonspatial = Input(shape=(NON_SPATIAL_INPUT_DIM,))



input2_spatial = Input(shape=(roman_seq_len, roman_vocab_size))

intm = LSTM(20)(input2_spatial)

intm = RepeatVector(ha_seq_len)(intm)

intm = LSTM(20, return_sequences=True)(intm)

intm = LSTM(20, return_sequences=True)(intm)

rnn_output = Flatten()(intm)



concatenated = keras.layers.concatenate([input1_nonspatial, rnn_output])



intm = Dense(ha_vocab_size*ha_seq_len*2)(concatenated)

intm = Activation('relu')(intm)



intm = Dense(ha_vocab_size*ha_seq_len)(intm)

intm = Reshape(target_shape=(ha_seq_len, ha_vocab_size))(intm)

output = Activation('softmax')(intm)



model = Model(inputs=[input1_nonspatial, input2_spatial], outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(x=[x_roman, x_roman_3], y=y_ha, epochs=1000, validation_split=0.3, batch_size=10)
y_pred = model.predict([x_roman,x_roman_3])
y_pred.shape
y_pred=y_pred.reshape((num_ex,ha_seq_len * ha_vocab_size))
NUM_PER_CHAR = ha_vocab_size

results = []

for i in (range(y_pred.shape[0])):

    char1_one_hot = y_pred[i,0:11]

    char1_encoded = np.argmax(char1_one_hot)

    char1 = ''

    if char1_encoded != 0:

        char1 = ha_index_to_char[char1_encoded]    

    

    char2_one_hot = y_pred[i,11:]

    char2_encoded = np.argmax(char2_one_hot)

    char2 = ''

    if char2_encoded != 0:

        char2 = ha_index_to_char[char2_encoded]

    results.append(char1 + char2)

    
for x,y in zip(list(roman_values), results):

    print(x,'\t',y)