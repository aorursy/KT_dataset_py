import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import h5py

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

from tensorflow.keras import datasets, layers, models, optimizers

# from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
tf.__version__
sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

train_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_set = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y_train_set = train_set["label"]

X_train_set = train_set.drop(labels = ["label"],axis = 1) 

del train_set
print(X_train_set.shape)

print(test_set.shape)
X_train_set = X_train_set.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

test_set = test_set.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0



Y_train_set = tf.keras.utils.to_categorical(Y_train_set, 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_set, Y_train_set, test_size = 0.1)
IMG_SIZE = (28, 28, 1)

input_img = layers.Input(shape=IMG_SIZE)

model = models.Sequential()

model = layers.Conv2D(32, (5, 5), padding='same', activation = 'relu')(input_img)

model = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation = 'relu')(model)



model = layers.MaxPooling2D((2, 2))(model)

model = layers.Dropout(0.25)(model)



model = layers.Conv2D(64, (3, 3), padding='same', activation = 'relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation = 'relu')(model)



model = layers.MaxPooling2D((2, 2), strides=(2, 2))(model)

model = layers.Dropout(0.25)(model)



model = layers.Flatten()(model)

# model = layers.GlobalAveragePooling2D()(model)

model = layers.Dense(256, activation = 'relu')(model)

model = layers.Dropout(0.25)(model)

model = layers.Dense(10)(model)

output_img = layers.Activation('softmax')(model)



model = models.Model(input_img, output_img)



model.summary()
adam = optimizers.Adam(lr=0.001) # The accuracy is at 97.98% if lr = 0.0001 

model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])



history = model.fit(X_train, Y_train, epochs=10, 

                    validation_data=(X_val, Y_val))
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.7, 1])

plt.legend(loc='best')
plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='best')
test_loss, test_accuracy = model.evaluate(X_val,  Y_val, verbose=2)



print('\nTest accuracy = {0:.2f}%'.format(test_accuracy*100.0))
# predict results

results = model.predict(test_set)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_tf2_digit.csv",index=False)