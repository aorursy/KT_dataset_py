import numpy as np

import tensorflow as tf
import gensim

model = gensim.models.Word2Vec.load('Twt-CBOW')
corpus_raw = 'He is the king . The king is royal . She is the royal queen'

corpus_raw = corpus_raw.lower()
words = []

for word in corpus_raw.split():

    if word != '.': 

        words.append(word)
words = set(words) 
words
word2int = {}

int2word = {}

vocab_size = len(words)



for i,word in enumerate(words):

    word2int[word] = i

    int2word[i] = word
print(word2int['queen'])
print(int2word[2])
raw_sentences = corpus_raw.split('.')

sentences = []

for sentence in raw_sentences:

    sentences.append(sentence.split())

                 
raw_sentences
print(sentences)
data = []

WINDOW_SIZE = 2



def to_one_hot(data_point_index, vocab_size):

    temp = np.zeros(vocab_size)

    temp[data_point_index] = 1

    return temp

for sentence in sentences:

    for word_index, word in enumerate(sentence):

        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 

            if nb_word != word:

                data.append([word, nb_word])
data
x_train = [] 

y_train = [] 

for data_word in data:

    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))

    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

x_train = np.asarray(x_train)

y_train = np.asarray(y_train)
print(x_train.shape, y_train.shape)
x_train[:5,:]
y_train[:5,:]
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(5, input_dim=x_train.shape[1]))

model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
history = model.fit(

    x_train,

    y_train,

    epochs=10000,

    shuffle=True,

    verbose=1,

)
weights_list = model.get_weights()

len(weights_list)
for i in range(len(weights_list)):

    print(weights_list[i].shape)
word2int['queen']
word2int
vectors
vectors = weights_list[0]

print(vectors[ word2int['queen'] ])
def euclidean_dist(vec1, vec2):

    return np.sqrt(np.sum((vec1-vec2)**2))



def find_closest(word_index, vectors):

    min_dist = 10000 # to act like positive infinity

    min_index = -1

    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):

        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):

            min_dist = euclidean_dist(vector, query_vector)

            min_index = index

    return min_index
print(int2word[find_closest(word2int['queen'], vectors)])