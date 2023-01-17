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
!pip install bnlp_toolkit
import nltk
nltk.download('punkt')
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
import re, string 
with open('/kaggle/input/converted2/converted.txt', 'r') as file:
    data = file.read()
    
data[:1000]
from bnlp.nltk_tokenizer import NLTK_Tokenizer
bnltk = NLTK_Tokenizer()

corpus = bnltk.sentence_tokenize(data)
corpus = corpus[1:]
print(len(corpus))
corpus = corpus[:10000]
corpus[935:955]
len(corpus)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(total_words)
input_sequences = []

for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n = token_list[:i+1]
    input_sequences.append(n)
len(input_sequences)
input_sequences = input_sequences[:100000]
len(input_sequences)
input_sequences[10000]
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)
print(total_words)
print(max_sequence_len)
print(len(n))
print(len(token_list))
a = [len(x) for x in input_sequences]
print(len(a))
print(len(input_sequences))
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Dropout, Bidirectional, SpatialDropout1D
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len))
model.add(Bidirectional(GRU(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(GRU(100))
# model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(predictors, label, epochs=50, batch_size= 256)
try:
	while(True):
		seed_text = input('Write something: \n')
		next_words = 10
		if seed_text == 'exit':
			break
		else:
			for _ in range(next_words):
				token_list = tokenizer.texts_to_sequences([seed_text])[0]
				token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
				predicted = model.predict_classes(token_list, verbose=0)
				output_word = ""
				for word, index in tokenizer.word_index.items():
					if index == predicted:
						output_word = word
						break
				seed_text += " " + output_word
			print(seed_text)
except:
	print('Error occured')