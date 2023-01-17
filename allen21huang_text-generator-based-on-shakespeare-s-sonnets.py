import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow.keras.utils as ku 

from wordcloud import WordCloud

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers
data = open('/kaggle/input/sonnets/sonnets.txt').read()

data[:500]
wordcloud = WordCloud(max_font_size=50, max_words=100,

                   background_color="white").generate(data)
plt.rcParams['savefig.dpi'] = 300

plt.figure(figsize = (8,4))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.savefig("poemcloud.png")

plt.show()
corpus = data.lower().split("\n")

corpus[:5]
length = []

for line in corpus:

    length.append(len(line))

pd.Series(length).describe()
corpus_new = []

for i in corpus:

    if len(i) == 0:

        continue

    else:

        corpus_new.append(i)
length_new = []

for line in corpus_new:

    length_new.append(len(line))

pd.Series(length_new).describe()
tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index)



print(total_words)
input_sequences = []

for line in corpus:

	token_list = tokenizer.texts_to_sequences([line])[0]

	for i in range(1, len(token_list)):

		n_gram_sequence = token_list[:i+1]

		input_sequences.append(n_gram_sequence)



max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))



predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words+1)
model = Sequential()

model.add(Embedding(total_words+1, 100, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(150, return_sequences = True)))

model.add(Dropout(0.2))

model.add(LSTM(100))

model.add(Dense(total_words+1/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(total_words+1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(predictors, label, epochs=150, verbose=1)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

loss = history.history['loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.title('Training accuracy')



plt.figure()



plt.plot(epochs, loss, 'b', label='Training Loss')

plt.title('Training loss')

plt.legend()



plt.show()
seed_text = "I love you, but"

next_words = 300

  

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