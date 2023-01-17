# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
# input data train dan test
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# pembagian data gambar dan label train
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
# normalisasi data gambar train dan test
X_train = X_train / 255.0
test = test / 255.0

print('Jumlah sampel gambar : {}'.format(X_train.shape[0]))
# reshaping gambar data
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# one hot encoding label
Y_train = to_categorical(Y_train, num_classes = 10)
# pembagian data train dan test untuk tiap gambar dan label
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=2)
# membuat model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', 
                         input_shape = (28,28,1)),
  tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  tf.keras.layers.Dropout(0.5),  
  tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'),
  tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation= 'relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation= 'softmax')
])
# menentukan optimizer
optimizer = Adam(lr=0.001)
# compile model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# membuat data generator
datagen = ImageDataGenerator(
        width_shift_range=0.25,
        brightness_range=[0.1,0.3],
        rotation_range=20,
        shear_range=0.20, 
        fill_mode='nearest',
        horizontal_flip=True,
        rescale=1./225,
)

datagen.fit(X_train)
class customCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs['val_accuracy'] > 0.96:
      self.model.stop_training = True
      print('Target val_accuracy di atas 96% sudah tercapai, training dihentikan!')

callback = customCallback()
# fitting model
batch_size = 76
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),
                              epochs = 5, validation_data = (X_val,Y_val), callbacks = [callback],
                              verbose = 2, steps_per_epoch = X_train.shape[0] // batch_size)
# membuat plot Loss untuk train/test dan accuracy untuk train/test dari model
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='lower right')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='lower right')

plt.show()
# Konversi model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)