from keras.datasets import imdb

from keras.preprocessing import sequence

from keras import layers

from keras.models import Sequential
max_features = 10000

maxlen = 500



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)



x_train = [x[::-1] for x in x_train]

x_test = [x[::-1] for x in x_test]



x_train = sequence.pad_sequences(x_train, maxlen = maxlen)

x_test = sequence.pad_sequences(x_test, maxlen = maxlen)
import matplotlib.pyplot as plt



def plot_results(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
model = Sequential()

model.add(layers.Embedding(max_features,128))

model.add(layers.LSTM(32))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop',

            loss = 'binary_crossentropy',

             metrics = ['accuracy'])



history = model.fit(x_train, y_train,

                   epochs = 10,

                   batch_size = 128,

                   validation_split = 0.2)
plot_results(history)
# Now we use a Bidirectional RNN



model = Sequential()

model.add(layers.Embedding(max_features,32))

model.add(layers.Bidirectional(layers.LSTM(32)))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop',

            loss = 'binary_crossentropy',

             metrics = ['accuracy'])



history = model.fit(x_train, y_train,

                   epochs = 10,

                   batch_size = 128,

                   validation_split = 0.2)
plot_results(history)