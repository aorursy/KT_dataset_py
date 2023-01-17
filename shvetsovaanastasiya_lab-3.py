import numpy as np

import matplotlib.pyplot as plt



import keras

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

batch_size = 128

num_classes = 10

epochs = 10

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols)



train_labels = keras.utils.to_categorical(train_labels, num_classes)

test_labels = keras.utils.to_categorical(test_labels, num_classes)



train_images = train_images/255.0

test_images = test_images/255.0



print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)
plt.figure (figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

#     print(train_images[i])

    plt.imshow(train_images[i])

    

plt.show()
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),

                activation = 'relu',

                input_shape=input_shape))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adadelta(),

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size= batch_size,

          epochs=epochs, verbose=1)
score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# from keras.datasets import cifar10

from tensorflow.python.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



num_classes = 10

batch_size = 128

epochs = 10

img_rows, img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



x_train = x_train / 255

x_test = x_test /255
plt.figure (figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

#     print(train_images[i])

    plt.imshow(x_train[i])

    

plt.show()
from keras import metrics

model_0 = Sequential()

model_0.add(Conv2D(32, kernel_size=(3,3),

                activation = 'relu',

                input_shape=input_shape))

model_0.add(Conv2D(64, (3,3), activation = 'relu'))

model_0.add(MaxPooling2D(pool_size=(2,2)))

model_0.add(Flatten())

model_0.add(Dense(128, activation = 'relu'))

model_0.add(Dense(num_classes, activation='softmax'))



model_0.compile(optimizer=keras.optimizers.Adadelta(),

              loss=keras.losses.categorical_crossentropy,

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics = ['accuracy'])

history_0 = model_0.fit(x_train, y_train, batch_size= batch_size,

          epochs=epochs, verbose=1, validation_split=0.1)

score = model_0.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def plot_val_loss(hist):

    plt.figure(figsize=plt.figaspect(0.5))

    l = range(0, len(hist.history['val_loss']))

    plt.plot(l, hist.history['val_loss'])

    plt.title('val_loss')

   
plot_val_loss(history_0)
model_1 = Sequential()

model_1.add(Conv2D(32, kernel_size=(3,3),

                activation = 'relu',

                input_shape=input_shape))

model_1.add(Conv2D(64, (3,3), activation = 'relu'))

model_1.add(MaxPooling2D(pool_size=(2,2)))

model_1.add(Flatten())

model_1.add(Dense(128, activation = 'relu'))

model_1.add(Dense(num_classes, activation='softmax'))



model_1.compile(optimizer=keras.optimizers.Adadelta(),

              loss=keras.losses.categorical_crossentropy,

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics=['accuracy'])

history_1 = model_1.fit(x_train, y_train, batch_size= batch_size,

          epochs=epochs, verbose=1, validation_split=0.25)

plot_val_loss(history_1)



score = model_1.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

epochs = 20



model_2 = Sequential()

model_2.add(Conv2D(32, kernel_size=(3,3),

                activation = 'relu',

                input_shape=input_shape))

model_2.add(Conv2D(32, (3,3), activation = 'relu'))

model_2.add(MaxPooling2D(pool_size=(2,2)))



model_2.add(Conv2D(64, (3, 3), activation = 'relu', input_shape=input_shape))

model_2.add(Conv2D(64, (3, 3), activation = 'relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2)))



model_2.add(Flatten())

model_2.add(Dense(256, activation = 'relu'))

model_2.add(Dense(num_classes, activation='softmax'))



model_2.compile(optimizer=keras.optimizers.Adadelta(),

              loss=keras.losses.categorical_crossentropy,

               metrics=['accuracy'])

history_2 = model_2.fit(x_train, y_train, batch_size= batch_size,

          epochs=epochs, verbose=1, validation_split=0.1)

plot_val_loss(history_2)



score = model_2.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
epochs = 40



model_3 = Sequential()

model_3.add(Conv2D(32, kernel_size=(3,3),

                activation = 'relu',

                input_shape=input_shape))

model_3.add(Conv2D(32, (3,3), activation = 'relu'))

model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Dropout(0.25))



model_3.add(Conv2D(64, (3, 3), activation = 'relu', input_shape=input_shape))

model_3.add(Conv2D(64, (3, 3), activation = 'relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Dropout(0.25))



model_3.add(Flatten())

model_3.add(Dense(256, activation = 'relu'))

model_3.add(Dropout(0.25))

model_3.add(Dense(num_classes, activation='softmax'))



model_3.compile(optimizer=keras.optimizers.Adadelta(),

              loss=keras.losses.categorical_crossentropy,

               metrics=['accuracy'])

history_3 = model_3.fit(x_train, y_train, batch_size= batch_size,

          epochs=epochs, verbose=1, validation_split=0.1)

plot_val_loss(history_3)



score = model_3.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
from sklearn.metrics import classification_report, confusion_matrix





y_pred = model_3.predict_classes(x_test)

y_test_cm = [np.argmax(y_test[i]) for i in range(0,y_test.shape[0])]



print(confusion_matrix(y_test_cm, y_pred))



print(classification_report(y_test_cm, y_pred))