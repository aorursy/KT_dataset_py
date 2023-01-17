from tensorflow.keras.datasets import imdb

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout,SimpleRNN

from tensorflow.keras import utils

from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 
max_words = 10000

maxlen    = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train[0]
x_train = pad_sequences(x_train, maxlen = maxlen, padding='post')

x_test = pad_sequences(x_test, maxlen = maxlen, padding='post')
x_train[0][:50]

len(x_train[0])

y_train[0]
max_len = 200

model = Sequential()

model.add(Embedding(max_words, 2, input_length=max_len))

model.add(SimpleRNN(8))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(x_train,

          y_train,

          epochs=15,

          validation_split=0.1)
plt.plot(history.history['accuracy'], 

         label='The share of correct answers on the training set')

plt.plot(history.history['val_accuracy'], 

         label='The share of correct answers on the checking set')

plt.xlabel('epochs')

plt.ylabel('Share of correct answers')

plt.legend()

plt.show()
scores = model.evaluate(x_test, y_test, verbose=1)
print("Percentage of correct answers on test data:", round(scores[1] * 100, 4))
embedding_matrix = model.layers[0].get_weights()[0]

embedding_matrix[:5]
#LOAD DICTIONARY WITH WORD'S INDEX

word_index_org = imdb.get_word_index()
# We supplement the dictionary with service characters



word_index = dict()

for word,number in word_index_org.items():

    word_index[word] = number + 3

word_index["<Заполнитель>"] = 0

word_index["<Начало последовательности>"] = 1

word_index["<Неизвестное слово>"] = 2  

word_index["<Не используется>"] = 3

word = 'good'

word_number = word_index[word]

print('Index of word', word_number)

print('Vector for word', embedding_matrix[word_number])
reverse_word_index = dict()

for key, value in word_index.items():

    reverse_word_index[value] = key
# We write dense vector representations to a file

class Vector_repr:

  def __init__(self):

    self.filename = 'imdb_embeddings.csv'

    with open(self.filename, 'w') as f:

      for word_num in range(max_words):

        self.word = reverse_word_index[word_num]

        self.vec = embedding_matrix[word_num]

        f.write(self.word + ",")

        f.write(','.join([str(x) for x in vec]) + "\n")

Vector_repr
plt.scatter(embedding_matrix[:,0], embedding_matrix[:,1])
# We select the word codes by which you can determine the tone of the recall



review = ['brilliant', 'fantastic', 'amazing', 'good',

          'bad', 'awful','crap', 'terrible', 'trash']

enc_review = []

for word in review:

    enc_review.append(word_index[word])

enc_review



review_vectors = embedding_matrix[enc_review]

review_vectors
plt.scatter(review_vectors[:,0], review_vectors[:,1])

for i, txt in enumerate(review):

    plt.annotate(txt, (review_vectors[i,0], review_vectors[i,1]))