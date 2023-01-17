import tensorflow as tf

from tensorflow import keras

import numpy as np
with open('../input/crime-and-punishment/crime_and_punishment.txt', encoding='utf-8') as f:

    lines = f.readlines()

    for i in range(0, len(lines)):

        lines[i] = lines[i].lower()

    

    # Let's separate the data based on sentences

    raw_sentences = list()

    for line in lines:

        for sentence in line.split('.'):

            raw_sentences.append(sentence)

    print(' --- Sentences before additional cleaning --- ')

    print(raw_sentences[:10])

    print(f'Number of sentences: {len(raw_sentences)}')

    

    # Those \ns sure are annoying...

    sentences = list()

    for i in range(0, len(raw_sentences)):

        raw_sentence = raw_sentences[i]

        clean_sentence = raw_sentence.split('\n')[0]

        if clean_sentence:

            sentences.append(clean_sentence)

            

    print(' --- Sentences after additional cleaning --- ')

    print(sentences[:10])

    print(f'Number of sentences: {len(sentences)}')        
# It's already time to tokenize! Let's check the total vocabulary size

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

print(len(tokenizer.word_index))
# We also need to make a 'labels' class

# create input sequences using list of tokens

max_sentence_length = 100

input_sequences = []

for sentence in sentences:

    token_list = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)



input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sentence_length+1, \

                                         padding='pre', truncating='post'))

word_index = tokenizer.word_index

num_tokens = len(word_index) + 1



predictors, label = input_sequences[:,:-1],input_sequences[:,-1]



label = keras.utils.to_categorical(label, num_classes=num_tokens)
print(len(label[0]))

print(len(predictors[0]))
# Let's use the GloVE word embeddings for this project, available here:

# nlp.stanford.edu/data/glove.6B.zip

# https://nlp.stanford.edu/pubs/glove.pdf

# Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.  GloVe: Global Vectors for Word Representation. 

embeddings_index = {}

with open('../input/glove-100d-word-embeddings/glove.6B.100d.txt', encoding = 'utf-8') as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, "f", sep=" ")

        embeddings_index[word] = coefs
# This code is adapted from here:

# https://keras.io/examples/nlp/pretrained_word_embeddings/

embedding_dim = 100

hits = 0

misses = 0



# Prepare embedding matrix

embedding_matrix = np.zeros((num_tokens, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # Words not found in embedding index will be all-zeros.

        # This includes the representation for "padding" and "OOV"

        embedding_matrix[i] = embedding_vector

        hits += 1

    else:

        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))

# That should do!

del(embeddings_index)
# Let's make the model, first editing the Embedding layer

from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(

    num_tokens,

    embedding_dim,

    embeddings_initializer=keras.initializers.Constant(embedding_matrix),

    input_length=max_sentence_length,

    trainable=True,

)
import tensorflow.keras.layers as layers

import tensorflow.keras.regularizers as regularizers

inputs = keras.Input(shape=(max_sentence_length,))

embedded_sequences = embedding_layer(inputs)

x = layers.Bidirectional(layers.LSTM(512, return_sequences = True))(embedded_sequences)

x = layers.Dropout(0.2)(x)

x = layers.LSTM(512, return_sequences=True)(x)

x = layers.Dropout(0.2)(x)

x = layers.LSTM(512)(x)

#x = layers.Dense(num_tokens/2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

preds = layers.Dense(num_tokens, activation = 'softmax')(x)

model = keras.Model(inputs, preds)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(predictors, label, batch_size=64, epochs=10, verbose=2)
# Using the model to generate some new text

seed_text = "upon"

next_words = 100



for _ in range(next_words):

  token_list = tokenizer.texts_to_sequences([seed_text])[0]

  token_list = pad_sequences([token_list], maxlen=max_sentence_length, padding='pre')

  predicted = np.argmax(model.predict(token_list, verbose=0), axis=1)

  output_word = ''

  for word, index in tokenizer.word_index.items():

    if index == predicted:

      output_word = word

      break

  seed_text += ' ' + output_word



print(seed_text)
# quite strange text, but it sure does sound like Dostoevsky!

# This text may differ from my GitHub and Colab variants