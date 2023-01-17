from __future__ import print_function



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.datasets import imdb



import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

MAX_FEATURES = 20000

# cut texts after this number of words (among top MAX_FEATURES most common words)

MAX_SENTENCE_LENGTH = 80



print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')



# View one example of our dataset before our preprocessing.

print("\n\nExample one before our preprocessing")

print(x_train[0])



print('\n\nPad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=MAX_SENTENCE_LENGTH)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_SENTENCE_LENGTH)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



# View one example of our dataset after our preprocessing.

print("\n\nExample one after our preprocessing")

print(x_train[0])

EMBEDDING_SIZE = 128

HIDDEN_LAYER_SIZE = 128

BATCH_SIZE = 32

NUM_EPOCHS = 15



print('Build model...')

model = Sequential()

model.add(Embedding(MAX_FEATURES, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))

model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



# summary of our model.

model.summary()



# Compile the model.

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print('Train...')

history = model.fit(x_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=NUM_EPOCHS,

          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,

                            batch_size=BATCH_SIZE)

print('Test score:', score)

print('Test accuracy:', acc)
plt.subplot(211)

plt.title("Accuracy")

plt.plot(history.history["accuracy"], color="g", label="Train")

plt.plot(history.history["val_accuracy"], color="b", label="Validation")

plt.legend(loc="best")

plt.subplot(212)

plt.title("Loss")

plt.plot(history.history["loss"], color="g", label="Train")

plt.plot(history.history["val_loss"], color="b", label="Validation")

plt.legend(loc="best")

plt.tight_layout()

plt.show()