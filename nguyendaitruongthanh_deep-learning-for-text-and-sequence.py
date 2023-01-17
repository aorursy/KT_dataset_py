# word-level one-hot encoding



from keras.preprocessing.text import Tokenizer 



samples = ["I love learning deep learning.",

           "Machine learning is the future of humanity."]



tokenizer = Tokenizer(num_words=1000) # only tokenize 1000 common words

tokenizer.fit_on_texts(samples)



sequences = tokenizer.texts_to_sequences(samples)



one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")



word_index = tokenizer.word_index

print("Found %s unique tokens." % len(word_index))
from keras.layers import Embedding



embedding_layer = Embedding(1000, 64) 

# max 1000 tokens/sequences, 64 dimensions/length
# IMDB movie-review sentiment-prediction



from keras.datasets import imdb

from keras import preprocessing

from keras.models import Sequential

from keras.layers import Flatten, Dense



max_features = 10000

maxlen = 20



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)



x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential()



model.add(Embedding(10000, 8, input_length=maxlen))



model.add(Flatten())



model.add(Dense(1, activation="sigmoid"))



model.summary()
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])



history = model.fit(x_train, y_train,

                   epochs=10,

                   batch_size=32,

                   validation_split=0.2)
# RNN in Keras



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, Dense



model = Sequential()

model.add(Embedding(10000, 32))

model.add(SimpleRNN(32, return_sequences=True))



model.summary()
from keras.datasets import imdb

from keras.preprocessing import sequence



max_features = 10000

maxlen = 500

batch_size = 32



print("Loading data...")

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(len(input_train), "train sequences")

print(len(input_test), "test sequences")





print("Pad sequences (samples x time)")

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)

input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print("input_train_shape: ", input_train.shape)

print("input_test_shape: ", input_test.shape)
model = Sequential()

model.add(Embedding(max_features, 32))

model.add(SimpleRNN(32))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])



history = model.fit(input_train, y_train,

                   epochs = 10,

                   batch_size = 128,

                   validation_split = 0.2)
import matplotlib.pyplot as plt



acc = history.history["acc"]

val_acc = history.history["val_acc"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, "b")

plt.plot(epochs, val_acc, "bo")

plt.title("Training and validation accuracy.")

plt.show()



plt.plot(epochs, loss, "r")

plt.plot(epochs, val_loss, "ro")

plt.title("Training and validation loss.")

plt.show()
from keras.layers import LSTM



model = Sequential()

model.add(Embedding(max_features, 32))

model.add(LSTM(32))

model.add(Dense(1, activation="sigmoid"))



model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])



history = model.fit(input_train, y_train,

                   epochs=10,

                   batch_size=128,

                   validation_split=0.2)
fname = "../input/weather-archive-jena/jena_climate_2009_2016.csv"

f = open(fname)



data = f.read()

f.close()



lines = data.split("\n")

header = lines[0].split(",")

lines = lines[1:]



print(header)

print(len(lines))
# convert into a NumPy array



import numpy as np



float_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):

    values = [float(x) for x in line.split(",")[1:]]

    float_data[i, :] = values
temp = float_data[:, 1]

plt.plot(range(len(temp)), temp)
plt.plot(range(1440), temp[:1440])
# to be updated