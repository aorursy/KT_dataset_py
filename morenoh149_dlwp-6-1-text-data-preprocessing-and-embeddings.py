import numpy as np
import os
from tqdm import tqdm
# Using tf for word-level one-hot encoding
from tensorflow.keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))
# Word-level one-hot encoding with hashing trick
# saves memory and allows online encoding
# (make sure dimensionality is larger than your vocabulary)
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality  # hashes word into random index between 0,1000
        results[i, j, index] = 1
results.shape
# we can learn more condensed representations of the data with Embeddings
from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(1000, 64)
embedding_layer.get_config()
# loading imdb data for use with an Embedding layer
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
max_features = 10000 # number of words
maxlen = 20 # top N words to use among the max_features most common words

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# using an embedding layer and classifier on imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
'''
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
                    '''
# using an embedding layer and classifier on imdb (Funtional api)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
inputs = Input(shape=(maxlen,))

x = Embedding(10000, 8)(inputs)
x = Flatten()(x)
predictions = Dense(1, activation='sigmoid')(x)

model_func = Model(inputs=inputs, outputs=predictions)
model_func.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model_func.summary()
'''
history = model_func.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
                    '''
# we will later add a Recurrent (lstm) or convolutional
# (conv) layer to reason about the words in context
# From raw text to word embeddings
!ls ../input/imdb-movie-reviews-dataset/aclimdb/aclImdb
!ls ../input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train
imdb_dir = '../input/imdb-movie-reviews-dataset/aclimdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in tqdm(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
# tokenize the raw text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 100 # cuts off reviews after 100 words
training_samples = 200 # trains on 200 samples to show how powerful GloVe is, otherwise task-specific trained embeddings will outperform GloVe
validation_samples = 10000 # validates on 10,000 samples
max_words = 10000 # considers only th top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# shuffle the data because the negative ones were loaded first
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]
# use fasttext word-embedding
!ls ../input/fasttext-wikinews
fasttext_dir = '../input/fasttext-wikinews'

embeddings_index = {}
f = open(os.path.join(fasttext_dir, 'wiki-news-300d-1M.vec'))
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found {} word vectors.'.format(len(embeddings_index)))
# preparing the FastText word-embeddings matrix
embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # words not found will be all zeroes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# load the fasttext embeddings into the model
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])
history_imdb_fasttext = model.fit(x_train, y_train,
                                 epochs=10,
                                 batch_size=32,
                                 validation_data=(x_val, y_val))
model.save_weights('pre_trained_fasttext_model.h5')
import matplotlib.pyplot as plt

acc = history_imdb_fasttext.history['acc']
val_acc = history_imdb_fasttext.history['val_acc']
loss = history_imdb_fasttext.history['loss']
val_loss = history_imdb_fasttext.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
# the model quickly overfits
# if we unfreeze the pretrained embedding we'll see worse performance as we don't
# have much data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_data=(x_val, y_val))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
# validation accuracy stalls in the low 50s
# if use more data however
# TODO try building a better model, the book says you should be able to do better by starting with
# the embeddings and training specifically for this task
maxlen = 100 # cuts off reviews after 100 words
training_samples = 2000 # trains on 2000 samples to show how powerful GloVe is, otherwise task-specific trained embeddings will outperform GloVe
validation_samples = 10000 # validates on 10,000 samples
max_words = 10000 # considers only th top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])
history_custom_embed = model.fit(x_train, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_data=(x_val, y_val))
acc = history_custom_embed.history['acc']
val_acc = history_custom_embed.history['val_acc']
loss = history_custom_embed.history['loss']
val_loss = history_custom_embed.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()