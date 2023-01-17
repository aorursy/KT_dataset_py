# Some prep for getting the dataset to work in Kaggle
from os import listdir, makedirs
from os.path import join, exists, expanduser

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
datasets_dir = join(cache_dir, 'datasets')
if not exists(datasets_dir):
    makedirs(datasets_dir)

# If you have multiple input files, change the below cp commands accordingly, typically:
# !cp ../input/keras-imdb/imdb* ~/.keras/datasets/
!cp ../input/imdb* ~/.keras/datasets/
from keras.datasets import imdb
from keras.preprocessing import sequence

max_words = 10000  # Our 'vocabulary of 10K words
max_len = 500  # Cut texts after 500 words

# Get data from Keras
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
word_index = imdb.get_word_index()
# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
embedding_dim = 100
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len)) # We train our own embeddings
model.add(Conv1D(32, 7, activation='relu')) # 1D Convolution, 32 channels, windows size 7
model.add(MaxPooling1D(5)) # Pool windows of size 5
model.add(Conv1D(32, 7, activation='relu')) # Another 1D Convolution, 32 channels, windows size 7
model.add(GlobalMaxPooling1D()) # Global Pooling
model.add(Dense(1)) # Final Output Layer

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=2,
                    batch_size=128,
                    validation_split=0.2)
from keras.layers import SimpleRNN
model = Sequential()
# No need to specify the sequence length anymore
model.add(Embedding(max_words, embedding_dim)) # We train our own embeddings
# RNN's only need their size as a parameter, just like Dense layers
model.add(SimpleRNN(32, activation='relu'))
# Dense output for final classification
model.add(Dense(1))

model.summary()
model = Sequential()
# No need to specify the sequence length anymore
model.add(Embedding(max_words, embedding_dim)) # We train our own embeddings
# This one returns the full sequence
model.add(SimpleRNN(32, activation='relu', return_sequences=True))
# This one just the last sequence element
model.add(SimpleRNN(32, activation='relu'))
# Dense output for final classification
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)
