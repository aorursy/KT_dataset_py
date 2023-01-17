import os, shutil
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
device_name = tf.test.gpu_device_name()
device_name

for dirname, _, fname in os.walk('../input'):
    print(dirname, '.........', len(os.listdir(dirname)))
image_size = 160
batch_size = 32

train_dir = '../input/mask-datasets-v1/Mask_Datasets/Train'
valid_dir = '../input/mask-datasets-v1/Mask_Datasets/Validation'

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    shear_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='binary'
    #class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='binary'
    #class_mode='categorical'
)
from keras.layers.normalization import BatchNormalization
IMG_SHAPE = (image_size, image_size, 3)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3),input_shape=(image_size, image_size, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.ReLU(),
  tf.keras.layers.MaxPooling2D((2, 2)),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(64, (3, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.ReLU(),
  tf.keras.layers.MaxPooling2D((2, 2)),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(64, (3, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.ReLU(),
  tf.keras.layers.MaxPooling2D((2, 2)),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(128, (3, 3)), 
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.ReLU(),
  tf.keras.layers.MaxPooling2D((2, 2)),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(256, (3, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.ReLU(),
  tf.keras.layers.GlobalAveragePooling2D(),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512),
  tf.keras.layers.Dense(1, activation='sigmoid')
  ])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics = ['accuracy'])

model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit(
    train_generator,
    epochs = 12,
    verbose=1,
    validation_data = valid_generator
)
model.save('model_normed.h5')
def plotting():
    epochs = np.arange(1,len(history.history['loss'])+1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(15,3))
    plt.subplot(121)
    plt.plot(epochs, train_acc, 'r', label='train' )
    plt.plot(epochs, val_acc, 'bo', label='valid')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.subplot(122)
    plt.plot(epochs, train_loss, 'r', label='train')
    plt.plot(epochs, val_loss, 'bo', label='valid')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

plotting()
!ls
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labelssoftmax.txt', 'w') as f:
  f.write(labels)
!ls
#saved_model_dir = 'model'
#tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelbase_normed.tflite', 'wb') as f:
  f.write(tflite_model)
!ls -l