%matplotlib inline

import matplotlib.pyplot as plt

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from keras.layers.advanced_activations import LeakyReLU
(x_train, y_train), (x_test, y_test) = mnist.load_data()



num_classes = 10

x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)





print('Train size:', x_train.shape[0])

print('Test size:', x_test.shape[0])
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(num_classes, activation='softmax'))



for l in model.layers:

    print(l.name, l.input_shape,'==>',l.output_shape)

print('\n')

print(model.summary())
batch_size = 128

epochs = 5



model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])

    

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=100)



print('\n')

print('Test loss:', round(score[0], 3))

print('Test accuracy:', round(score[1], 3))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
epochs = 20



for activation in [None, 'sigmoid', 'tanh', 'relu']:

    model = Sequential()

    model.add(Dense(512, activation=activation, input_shape=(784,)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])

    history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))

    

    plt.plot(history.history['val_accuracy'])

    

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['None', 'sigmoid', 'tanh', 'relu'], loc='upper left')