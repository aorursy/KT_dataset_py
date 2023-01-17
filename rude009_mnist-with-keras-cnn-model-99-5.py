import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

import tensorflow as tf



from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model

from keras.callbacks import EarlyStopping



pd.set_option('display.max_rows', 1000)

warnings.filterwarnings("ignore")



dftrain = pd.read_csv('../input/digit-recognizer/train.csv')

dftest = pd.read_csv('../input/digit-recognizer/test.csv')
IMG_SIZE = 28



x_train = dftrain.iloc[:,1:]

x_train = x_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = dftrain.iloc[:,0]

x_test = dftest

x_test = x_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x_train = x_train/255.0

x_test = x_test/255.0
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1)

datagen.fit(X_train)
earlystopping = EarlyStopping(monitor ="val_accuracy",

                              mode = 'auto', patience = 30,

                              restore_best_weights = True)







model = Sequential()



model.add(Conv2D(128, (3, 3), input_shape = x_train.shape[1:]))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Dropout(0.2))



model.add(Conv2D(512, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Dropout(0.2))





model.add(Flatten())





model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))





model.add(Dense(10, activation='softmax'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(optimizer = 'adam',

  loss = 'sparse_categorical_crossentropy',

  metrics = ['accuracy'])



EPOCHS = 1000

BATCH_SIZE=64



history = model.fit(datagen.flow(X_train, y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[earlystopping])
print("Max. Validation Accuracy: {}%".format(round(100*max(history.history['val_accuracy']), 2)))
fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

loss_train = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1,len(loss_val)+1)

plt.plot(epochs, loss_train, 'g', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='Validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.subplot(1, 2, 2)

acc_train = history.history['accuracy']

acc_val = history.history['val_accuracy']

epochs = range(1,len(acc_val)+1)

plt.plot(epochs, acc_train, 'g', label='Training accuracy')

plt.plot(epochs, acc_val, 'b', label='Validation accuracy')

plt.title('Training and Validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictions = model.predict([x_test])

solutions = []

for i in range(len(predictions)):

    solutions.append(np.argmax(predictions[i]))
final = pd.DataFrame()

final['ImageId']=[i+1 for i in dftest.index]

final['Label']=solutions

final.to_csv('submission.csv', index=False)