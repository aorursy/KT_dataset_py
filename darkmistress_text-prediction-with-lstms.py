

from keras.preprocessing.text import Tokenizer

from keras.layers import Embedding, LSTM, Dense

from keras.models import Sequential

import pandas as pd

import numpy as np





text = 'It is not the strength of the body but the strength of the spirit. It is useless to meet revenge with revenge it will heal nothing. Even the smallest person can change the course of history. All we have to decide is what to do with the time that is given us. The burned hand teaches best. After that, advice about fire goes to the heart'



# Split text into an array of words

words = text.split()



# Make sentences of 4 words each, moving one word at a time

sentences = []

for i in range(4, len(words)):

  sentences.append(' '.join(words[i-4:i]))

  

# Instantiate a Tokenizer, then fit it on the sentences

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)



# Turn sentences into a sequence of numbers

sequences = tokenizer.texts_to_sequences(sentences)

print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))



model = Sequential()



vocab_size = len(tokenizer.index_word) + 1



# Add an Embedding layer with the right parameters

model.add(Embedding(input_dim = vocab_size, input_length = 3, output_dim = 8))



# Add a 32 unit LSTM layer

model.add(LSTM(32))



# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax

model.add(Dense(32, activation='relu'))

model.add(Dense(vocab_size, activation='softmax'))

model.summary()
def predict_text(test_text, model = model):

    if len(test_text.split()) != 3:

        print('Text input should be 3 words!')

        return False

  

  # Turn the test_text into a sequence of numbers

    test_seq = tokenizer.texts_to_sequences([test_text])

    test_seq = np.array(test_seq)

  

  # Use the model passed as a parameter to predict the next word

    pred = model.predict(test_seq).argmax(axis = 1)[0]

   

  # Return the word that maps to the prediction

    return tokenizer.index_word[pred]
predict_text('to do with')