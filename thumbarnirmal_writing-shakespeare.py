from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Embedding

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow.keras.utils as ku

import numpy as np
tokenizer = Tokenizer()



data = open('../input/sonnets.txt').read()



corpus = data.lower().split('\n')



tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1



input_sequences = []

for line in corpus:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(2, len(token_list)+1):

        n_gram_sequence = token_list[:i]

        input_sequences.append(n_gram_sequence)



max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))



predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]



labels = ku.to_categorical(labels, num_classes=total_words)
model = Sequential()

model.add(Embedding(total_words, 256, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(256, return_sequences=True)))

model.add(Dropout(0.2))

model.add(LSTM(512))

model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2()))

model.add(Dense(total_words, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
checkpoint_path = '/kaggle/working/model.ckpt'



cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
#history = model.fit(predictors, labels, epochs=100, verbose=1, callbacks=[cp_callback])
# To run model from saved weights

'''

model = Sequential()

model.add(Embedding(total_words, 256, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(256, return_sequences=True)))

model.add(Dropout(0.2))

model.add(LSTM(512))

model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2()))

model.add(Dense(total_words, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())



checkpoint_path = '/kaggle/working/model.ckpt'



model.load_weights(checkpoint_path)

'''
import matplotlib.pyplot as plt

acc = history.history['acc']

loss = history.history['loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.title('Training accuracy')



plt.figure()



plt.plot(epochs, loss, 'b', label='Training Loss')

plt.title('Training loss')

plt.legend()



plt.show()
seed_text = "Help me Obi Wan Kenobi, you're my only hope"

next_words = 100

  

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted = model.predict_classes(token_list, verbose=0)

    output_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            output_word = word

            break

    seed_text += " " + output_word

print(seed_text)