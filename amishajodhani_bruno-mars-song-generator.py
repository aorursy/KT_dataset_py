# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow
print(tensorflow.__version__)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np
data = open('/kaggle/input/poetry/bruno-mars.txt').read()

tokenizer = Tokenizer()

corpus = data.lower().split("\n")


tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  #adding one for out-of-vocab words

# creating input sequences using list of tokens
input_sequences = []
for l in corpus:
	token_list = tokenizer.texts_to_sequences([l])[0]
	for i in range(1, len(token_list)):
		n_gram_seq = token_list[:i+1]
		input_sequences.append(n_gram_seq)


# implementing padding
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

# one hot code
label = ku.to_categorical(label, num_classes=total_words)
model= Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1), # -1 as the last word is the label
    Bidirectional(LSTM(150, return_sequences = True)),
    Dropout(0.2),  #drops 20% of units in a layer
    LSTM(100),
    Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
num_epochs= 100
history = model.fit(predictors, label, epochs=num_epochs, verbose=1)

# LAST EPOCH I GOT:
#Epoch 100/100
#739/739 [==============================] - 29s 39ms/step - loss: 1.0005 - accuracy: 0.8041
import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()
start_text = "You know what I want"
num_words = 60  #number of words to generate
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([start_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	out_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			out_word = word
			break
	start_text += " " + out_word
print(start_text)