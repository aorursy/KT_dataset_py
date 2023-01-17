### Adapted from: https://youtu.be/zwdfUIhpIQo
### Subscribe my channel on YouTube: https://www.youtube.com/RiteshKumarMaurya

#IDEA:
"""The idea is to build a model which is trained on a series of song lyrics and then use it to generate further lyrics with some input.
For example, the model will take a small string from the user (Like: Happy Birthday to you) and
then it will try to generate the lyrics starting from this line!
For this purpose, I'm using a Keras based RNN, i.e., Recurrent Neural Network with Long Short-Term Memory(LSTM) for genrating lyrics,
which will differ in each case and will also memorize or learns from the past predictions.

I'm using Keras with Tensorflow as backened. Keras lets us to create and solve our problems in an object oriented way.
So, there are basically two dependencies for this kernel, viz., Keras & Tensorflow.

"""

# There is also a program named "Helper.py" included in the kernel as data, which is used for performing some very important functions.
# So, let's get started.
# Gathering Lyric data from "55000+ Song Lyrics" and convrting it into plain text for training the model.

import csv

#Opening the file (SONGDATA.csv) for extracting the text. 
with open('../input/songlyrics/songdata.csv', 'r') as file:

    reader = csv.reader(file)

    # Opening DATA.py file for writing the training data in it.
    corpus = open('lyrics_data.txt', 'a')

    for row in reader:
        # Data Structure.
        #print (row)
        text = row[3]
        corpus.write(text) 
import os
os.listdir()

# Testing our data.txt file:
file = open('lyrics_data.txt', 'r')
print (len(file.read())) #67995497
# Now we have a huge amount of data
#Loading the modules:

from __future__ import print_function
import numpy as np
import random
from keras.models import load_model
import argparse
import sys

#Importing helper.py program
sys.path.insert(0, '../input/ailyricshelper')
import helper
"""
    Define global variables.
"""
SEQUENCE_LENGTH = 40
SEQUENCE_STEP = 3
PATH_TO_CORPUS = "lyrics_data.txt"
EPOCHS = 25
DIVERSITY = 1.0

"""
    Read the corpus and get unique characters from the corpus.
"""
text = helper.read_corpus(PATH_TO_CORPUS)
chars = helper.extract_characters(text)
"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
sequences, next_chars = helper.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)
char_to_index, indices_char = helper.get_chars_index_dicts(chars)

"""
    The network is not able to work with characters and strings, we need to vectorise.
"""
X, y = helper.vectorize(sequences, SEQUENCE_LENGTH, chars, char_to_index, next_chars)

"""
    Define the structure of the model.
"""
model = helper.build_model(SEQUENCE_LENGTH, chars)


#    If you want to Train the model, uncomment this line.
# model.fit(X, y, batch_size=128, nb_epoch=EPOCHS)
#model.save('final_model.h5')


#   If you want to test and see demo of the model, keep this line uncommented.
model = load_model("../input/modelfile/model.h5")  # you can skip training by loading the trained weights

"""
    Pick a random sequence and make the network continue
"""

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('Diversity:', diversity)

    generated = ''
    
    sentence = "I will wash away your pain with my tears"

    sentence = sentence.lower()
    generated += sentence

    print('Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(500):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = helper.sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
