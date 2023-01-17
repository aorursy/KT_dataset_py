from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from tensorflow.keras.models import model_from_json



import numpy as np

print(np.__version__)

import math
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
# [print('{}\n'.format(X_train[i])) for i in range(len(X_train)) if i < 5]



print('=== Data ====\n{}\n\n=== Sentiment ====\n{}\n\n{}'.format(

    X_train[5], y_train[5], type(X_train)))
# Default value in load_data

INDEX_FROM = 3



# Download word index and prepare word id

word2id = imdb.get_word_index()

word2id = {word:(word_id + INDEX_FROM) for (word, word_id) in word2id.items()}

# Labelling predefined value to prevent error

word2id["<PAD>"] = 0

word2id["<START>"] = 1

word2id["<UNK>"] = 2

word2id["<UNUSED>"] = 3



# Prepare id to word

id2word = {value:key for key, value in word2id.items()}



print('=== Tokenized sentences words ===')

print(' '.join(id2word[word_id] for word_id in X_train[5]))
pad_size = 1000

X_train_pad = pad_sequences(X_train, maxlen=pad_size)

X_test_pad = pad_sequences(X_test, maxlen=pad_size)
vocab_size = len(word2id)

input_dim = math.ceil(vocab_size / 2)

print('Len Vocab: {}, input_dim: {}'.format(len(word2id), input_dim))

embedding_size = math.ceil(vocab_size**0.25)

output_units = 1



model=Sequential()

model.add(Embedding(input_dim=input_dim, output_dim=embedding_size,

    input_length=pad_size))

model.add(LSTM(units=100))

model.add(Dense(output_units, activation='sigmoid'))



model.compile(loss='binary_crossentropy', 

             optimizer='adam', 

             metrics=['accuracy'])



print(model.summary())
train_size = math.ceil(0.8 * len(X_train))



X, y = X_train_pad[:train_size], y_train[:train_size]

X_val, y_val = X_train_pad[train_size:], y_train[train_size:]
batch_size = 64

epochs = 5



model.fit(X, y, validation_data=(X_val, y_val), batch_size=batch_size,

          epochs=epochs, shuffle=True)
# Saving structure and weights

model_structure = model.to_json()

with open('model_structure.json', 'w') as f:

    f.write(model_structure)

    

model.save_weights('model_weights.h5')
# Load and compile model



with open('model_structure.json', 'r') as f:

    loaded_model_json = f.read()

    loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('model_weights.h5')

loaded_model.compile(loss='binary_crossentropy', 

             optimizer='adam', 

             metrics=['accuracy'])
scores = loaded_model.evaluate(X_test_pad, y_test, verbose=0)

print('Model Accuracy:', scores[1])