%matplotlib inline

import warnings

import pandas as pd

import numpy as np

from keras.datasets import cifar10

from keras.models import Sequential

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.models import load_model

from keras.layers import (

    Dense,

    Flatten,

    Dropout,

    Conv2D,

    MaxPooling2D,

    Activation,

    BatchNormalization

)

from keras.utils import np_utils

from keras.preprocessing import image

import matplotlib.pyplot as plt

from PIL import Image

from keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
def plot_history(history):

    history_dict = history.history

    loss_values = history_dict['loss']

    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)





    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))



    ax1.plot(epochs, loss_values, 'bo',

             label='Training loss')

    ax1.plot(epochs, val_loss_values, 'r',

             label='Validation loss')

    ax1.set_xlabel('Epochs')

    ax1.set_ylabel('Loss')

    ax1.set_xscale('log')



    acc_values = history_dict['accuracy']

    val_acc_values = history_dict['val_accuracy']



    ax2.plot(epochs, acc_values, 'bo',

             label='Training acc')

    ax2.plot(epochs, val_acc_values, 'r',

             label='Validation acc')

    ax2.set_xlabel('Epochs')

    ax2.set_ylabel('Accuracy')

    ax2.set_xscale('log')





    plt.legend()

    plt.show()
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
X_train = train.loc[:, train.columns!='label'].values.astype('uint8')

y_train = train['label'].values



X_train = X_train.reshape((X_train.shape[0],28,28))
X_test = test.loc[:, test.columns!='label'].values.astype('uint8')

X_test = X_test.reshape((X_test.shape[0],28,28))
n = np.random.randint(X_train.shape[0])

plt.imshow(Image.fromarray(X_train[n]))

plt.show()

print(f'This is a {y_train[n]}')
X_train = X_train[:,:,:,None]

X_test = X_test[:,:,:,None]
X_train.shape
batch_size = 32

num_samples = X_train.shape[0]

num_classes = np.unique(y_train).shape[0]

num_epochs = 50

img_rows, img_cols = X_train[0,:,:,0].shape

img_channels = 1

classes = np.unique(y_train)
y_train = np_utils.to_categorical(y_train, num_classes)

# y_test = np_utils.to_categorical(y_test, num_classes)
X_train_norm = X_train.astype('float32')

X_test_norm = X_test.astype('float32')

X_train_norm /= 255

X_test_norm /= 255
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=5,

                                            verbose=1,

                                            factor=0.2)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history)
! mkdir newer
model.save('newer/simple.h5')
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history1 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history1)
model.save('newer/simple_batch.h5')
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(64, kernel_size=(3, 3),

                 activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history2)
model.save('newer/32x64_64x128.h5')
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (5, 5), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history3 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history3)
model.save('newer/32x64x64.h5')
model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history4 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history4)
model.save('newer/64x128.h5')
model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(256, kernel_size=(3, 3),

                 activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history5 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history5)
model.save('newer/64x128_256x512.h5')
model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(256, kernel_size=(3, 3),

                 activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history6 = model.fit(

    X_train_norm,

    y_train,

    batch_size=batch_size,

    epochs=num_epochs,

    validation_split=0.1,

    shuffle=True,

    callbacks=[learning_rate_reduction, es]

)
plot_history(history6)
model.save('newer/64x128_256x512_diff_fcnn.h5')
labels = ['simple', 'simple+batch', '32+64_64+128',

          '32+64+64', '64+128', '64+128_256+512', '64+128_256+512_diff_fcnn']

plt.figure(figsize=(7.5,7.5))

for idx, h in enumerate([history, history1, history2,

          history3, history4, history5, history6]):

    history_dict = h.history

    loss_values = history_dict['loss']

    epochs = range(1, len(loss_values) + 1)

    

    plt.plot(epochs, loss_values, label=labels[idx],

             linestyle='-', marker='o')



plt.title('Training loss function')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend()

plt.show()





plt.figure(figsize=(7.5,7.5))

for idx, h in enumerate([history, history1, history2,

          history3, history4, history5, history6]):

    history_dict = h.history

    loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    

    plt.plot(epochs, loss_values, label=labels[idx],

             linestyle='-', marker='o')



plt.title('Validation loss function')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend()

plt.show()
labels = ['simple', 'simple+batch', '32+64_64+128',

'32+64+64', '64+128', '64+128_256+512', '64+128_256+512_diff_fcnn']

plt.figure(figsize=(7.5,7.5))

for idx, h in enumerate([history, history1, history2,

          history3, history4, history5, history6]):

    history_dict = h.history

    acc_values = history_dict['accuracy']

    epochs = range(1, len(acc_values) + 1)

    

    plt.plot(epochs, acc_values, label=labels[idx],

             linestyle='-', marker='o')



plt.title('Training accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend()

plt.show()





plt.figure(figsize=(7.5,7.5))

for idx, h in enumerate([history, history1, history2,

          history3, history4, history5, history6]):

    history_dict = h.history

    acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)

    

    plt.plot(epochs, acc_values, label=labels[idx],

             linestyle='-', marker='o')



plt.title('Validation accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend()

plt.show()
model = load_model('newer/64x128_256x512_diff_fcnn.h5')
pred = model.predict_classes(X_test_norm)
sample_submission['Label'] = pred
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)