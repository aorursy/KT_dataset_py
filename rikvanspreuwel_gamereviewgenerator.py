import numpy as np

import pandas as pd

import os

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.layers import RNN

from keras.utils import np_utils

import re

from langdetect import detect

import matplotlib.pyplot as plt
metacritic_game_user_comments = pd.read_csv("../input/metacritic-video-game-comments/metacritic_game_user_comments.csv")

metacritic_game_user_comments.head()
xc2_reviews = metacritic_game_user_comments[(metacritic_game_user_comments['Title'] == 'Xenoblade Chronicles 2')]

xc2_reviews = xc2_reviews[(xc2_reviews['Userscore'] == 10)]

print(xc2_reviews.shape)
# Print the last row to view the noice of the "This review contains spoilers.... " line

# And remove that part of the strings containing the substring

print(xc2_reviews.tail(1))

xc2_reviews['Comment'] = xc2_reviews['Comment'].str.replace('            This review contains spoilers, click expand to view.        ', '')



# Detect language and select only English reviews.

xc2_reviews['Language'] = xc2_reviews['Comment'].apply(detect)

xc2_reviews = xc2_reviews[(xc2_reviews['Language'] == 'en')]['Comment']

print(xc2_reviews.shape)
# Convert all column values into one lowercase string

xc2_reviews_string = '\n'.join(xc2_reviews.values).lower()

print(xc2_reviews_string[:500])
# to count words in string 

res = len(re.findall(r'\w+', xc2_reviews_string)) 

res
characters = sorted(list(set(xc2_reviews_string)))



n_to_char = {n:char for n, char in enumerate(characters)}

char_to_n = {char:n for n, char in enumerate(characters)}
X = []

Y = []

input_length = len(xc2_reviews_string)

seq_length = 100



# Loop through the entire input string and create sequences of characters

for i in range(0, input_length - seq_length, 1):

    sequence = xc2_reviews_string[i:i + seq_length]

    label = xc2_reviews_string[i + seq_length]

    

    X.append([char_to_n[char] for char in sequence])

    Y.append(char_to_n[label])



print("Total Patterns:", len(X))
X_modified = np.reshape(X, (len(X), seq_length, 1))

X_modified = X_modified / float(len(characters))

Y_modified = np_utils.to_categorical(Y)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.layers import RNN



model = Sequential()

model.add(LSTM(600, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(600))

model.add(Dropout(0.2))

model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
history = model.fit(X_modified, Y_modified, epochs=100, batch_size=100)

model.save_weights('xc2_review_generator_model_with_bigger_layers.h5')
# https://keras.io/visualization/

# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train'], loc='upper left')

plt.show()
def combine_string(character_array):

    return_string = ""

    for char in character_array:

        return_string = return_string + char

    return return_string



def generate_review(starting_sequence):

    generated_review_mapped = [n_to_char[value] for value in starting_sequence]

    print(f'Starting sequence: {combine_string(generated_review_mapped)}')

    

    for i in range(400):

        x = np.reshape(starting_sequence,(1,len(starting_sequence), 1))

        x = x / float(len(characters))



        pred_index = np.argmax(model.predict(x, verbose=0))

        seq = [n_to_char[value] for value in starting_sequence]

        generated_review_mapped.append(n_to_char[pred_index])



        starting_sequence.append(pred_index)

        starting_sequence = starting_sequence[1:len(starting_sequence)]



    return generated_review_mapped
generated_review_1 = generate_review(X[99].copy())

print(combine_string(generated_review_1))
generated_review_2 = generate_review(X[0].copy())

print(combine_string(generated_review_2))
generated_review_3 = generate_review(X[10].copy())

print(combine_string(generated_review_3))
custom_starting_sequence = list('xenoblade chronicles 2 is a great game, like all games, it has its issues, but it runs smoothly and ')

custom_starting_sequence_mapped = [char_to_n[value] for value in custom_starting_sequence]



generated_custom_review = generate_review(custom_starting_sequence_mapped)

print(combine_string(generated_custom_review))