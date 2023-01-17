import string

import numpy as np

from keras.preprocessing.text import Tokenizer
# This is our initial data; one entry per "sample"

samples = ['Rikki-tikki knew better than to waste time in staring.','He jumped up in the air as high as he could go.']



# First, build an index of all tokens in the data.

token_index = {}

for sample in samples:

    # We simply tokenize the samples via the `split` method.

    # in real life, we would also strip punctuation and special characters

    for word in sample.split():

        if word not in token_index:

            # Assign a unique index to each unique word

            token_index[word] = len(token_index) + 1



# vectorize our samples.

# We will only consider the first `max_length` words in each sample.

max_length = 10



# This is where we store our results:

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):

    for j, word in list(enumerate(sample.split()))[:max_length]:

        index = token_index.get(word)

        results[i, j, index] = 1.
samples = ['Rikki-tikki knew better than to waste time in staring.','He jumped up in the air as high as he could go.']

characters = string.printable  # All printable ASCII characters.

token_index = dict(zip(characters, range(1, len(characters) + 1)))



max_length = 50

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):

    for j, character in enumerate(sample[:max_length]):

        index = token_index.get(character)

        results[i, j, index] = 1.
samples = ['Rikki-tikki knew better than to waste time in staring.','He jumped up in the air as high as he could go.']



# We create a tokenizer, configured to only take

# into account the top-1000 most common words

tokenizer = Tokenizer(num_words=1000)

# This builds the word index

tokenizer.fit_on_texts(samples)



# This turns strings into lists of integer indices.

sequences = tokenizer.texts_to_sequences(samples)



one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')



# This is how you can recover the word index that was computed

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
samples = ['Rikki-tikki knew better than to waste time in staring.','He jumped up in the air as high as he could go.']

dimensionality = 1000

max_length = 10



results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):

    for j, word in list(enumerate(sample.split()))[:max_length]:

        index = abs(hash(word)) % dimensionality

        results[i, j, index] = 1.