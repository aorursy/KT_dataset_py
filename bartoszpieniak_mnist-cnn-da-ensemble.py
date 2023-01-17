import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split



import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_train_raw = train.drop(columns=['label']).to_numpy()

y_train_raw = train['label'].to_numpy()
print(X_train_raw.shape)

print(y_train_raw.shape)
x_mean = np.mean(X_train_raw)

x_std = np.std(X_train_raw)



def standarize(X):

    return (X - x_mean) / x_std



X_train = np.reshape(X_train_raw,(-1,28,28,1))

X_train = standarize(X_train)



y_train_cat = keras.utils.to_categorical(y_train_raw)
y_train_cat.shape
data_generator = keras.preprocessing.image.ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1

)



data_generator.fit(X_train)
def make_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=3,activation='relu',kernel_initializer='he_normal',input_shape=(28, 28, 1)))

    model.add(keras.layers.BatchNormalization()) 

    model.add(keras.layers.Conv2D(32, kernel_size=3,activation='relu'))

    model.add(keras.layers.BatchNormalization()) 

    model.add(keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',activation='relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.4))

    

    model.add(keras.layers.Conv2D(64, kernel_size=3, activation ='relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, kernel_size=3, activation ='relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',activation='relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.4))

    

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation = "relu"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(10, activation = "softmax"))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
ens_size=16

model = [0]*ens_size



for i in range(ens_size):

    model[i] = make_model()
import time

history = [0]*ens_size





start = time.perf_counter()



for i in range(ens_size):

    callbacks = [keras.callbacks.ModelCheckpoint('/kaggle/working/mdl-{}-of-{}.hdf5'

                                                 .format(i,ens_size-1),save_best_only=True, monitor='val_accuracy', mode='max')]

    

    X_train_ens, X_valid_ens, y_train_ens, y_valid_ens = train_test_split(X_train, y_train_cat, train_size=0.9)

    train_aug = data_generator.flow(X_train_ens, y_train_ens)

    

    history[i] = model[i].fit_generator(train_aug, epochs=40,

                              validation_data=(X_valid_ens, y_valid_ens),

                              callbacks=callbacks, verbose = 0)

    

    print("CNN {}: Train accuracy={:0.5f}, Validation accuracy={:0.5f}"

      .format(i,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))

    

stop = time.perf_counter()

print(f"{stop - start:0.4f}")
import matplotlib.pyplot as plt



def visualize_accuracy(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper right')

    plt.show()



def visualize_loss(history):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper right')

    plt.show()

    

visualize_accuracy(history[0])

visualize_loss(history[0])
X_test = np.reshape(test.to_numpy(),(-1,28,28,1))

X_test = standarize(X_test)
preds = np.zeros((X_test.shape[0],10))



for i in range(ens_size):

    mdl = keras.models.load_model('/kaggle/working/mdl-{}-of-{}.hdf5'.format(i,ens_size-1))

    preds = preds + mdl.predict(X_test)
submit_pred = np.argmax(preds,axis=1)

submit_pred.shape
submition = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submition['Label'] = submit_pred
submition.to_csv('/kaggle/working/submition_ens3.csv', index=False)