import numpy as np 

import pandas as pd 

import re

from keras.models import Model

from keras.layers import Input, Dense, Reshape, merge

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import skipgrams

from keras.preprocessing import sequence

from keras.layers import dot

from keras.models import Sequential

from keras.preprocessing import text

import collections
from nltk.corpus import brown

from nltk.corpus import stopwords
def preprocess(corpus):

    text = [re.sub("[^A-Za-z']+", '', word).lower() for word in corpus]

    text = [re.sub('[\']{2}', '', word) for word in text]

    text = [word for word in text  if not word == '']



    return text



training_data = preprocess(brown.words())
def create_voc(words):

    count = [['UNK', 0]]

    count.extend(collections.Counter(words).most_common(len(words) - 1))

    dictionary = {}

    for word, _ in count:

        dictionary[word] = len(dictionary)

    data = []

    unk_count = 0

    for word in words:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0  # dictionary['UNK']

            unk_count += 1

        data.append(index)

    count[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary  



data, count, dictionary, reverse_dictionary = create_voc(training_data)
window_size = 3

vector_dim = 100
vocab_size = len(dictionary)

sampling_table = sequence.make_sampling_table(vocab_size)

couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)

word_target, word_context = zip(*couples)

word_target = np.array(word_target, dtype="int32")

word_context = np.array(word_context, dtype="int32")

input_target = Input((1,))

input_context = Input((1,))



embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)

target = Reshape((vector_dim, 1))(target)

context = embedding(input_context)

context = Reshape((vector_dim, 1))(context)



dot_product = dot([target, context], axes=1, normalize=False)

dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(inputs=[input_target, input_context], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.summary()
# arr_1 = np.zeros((1,))

# arr_2 = np.zeros((1,))

# arr_3 = np.zeros((1,))

# iters = 100000

# for cnt in range(iters):

#     idx = np.random.randint(0, len(labels)-1)

#     arr_1[0,] = word_target[idx]

#     arr_2[0,] = word_context[idx]

#     arr_3[0,] = labels[idx]

#     loss = model.train_on_batch([arr_1, arr_2], arr_3)

#     if cnt % 1000 == 0:

#         print("Iteration {}, loss={}".format(cnt, loss))

 
# pd.DataFrame(weights, index=reverse_dictionary.values()).to_csv('out.csv', sep='\t')
weights = model.layers[2].get_weights()[0][1:]



print(weights.shape)

reverse_dictionary.pop(47138)

pd.DataFrame(weights, index=reverse_dictionary.values()).head(25)

weights_csv = pd.read_csv('../input/word2vec-embeddings/out.csv') 
from sklearn.metrics.pairwise import cosine_similarity



distance_matrix = cosine_similarity(weights)

print(distance_matrix.shape)

similar_words = {search_term: [reverse_dictionary[idx] for idx in distance_matrix[dictionary[search_term]-1].argsort()[-6:-1]+1] 

                   for search_term in ['thief', 'monster', 'bank', 'mars']}

similar_words