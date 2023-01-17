import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as implt
from tensorflow.keras.applications.inception_v3 import InceptionV3
train_dir = "/kaggle/input/horses-or-humans-dataset/horse-or-human/train"
test_dir = "/kaggle/input/horses-or-humans-dataset/horse-or-human/validation"

train_humans = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/humans")
train_horses = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/horses")

test_humans = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/validation/humans")
test_horses = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/validation/horses")
print("Number of images in the train-set:", len(train_horses) + len(train_humans))
print("Number of images in the test-set:", len(test_horses) + len(test_humans))

print("\nNumber of humans in the train-set:", len(train_humans))
print("Number of horses in the train-set:", len(train_horses))

print("\nNumber of humans in the test-set:", len(test_humans))
print("Number of horses in the test-set:", len(test_horses))
import random

fig, ax = plt.subplots(2,4, figsize=(15, 8))
for i in range(4):
    x = random.randint(0, len(train_horses))
    ax[0, i].imshow(implt.imread(train_path + '/humans/' + train_humans[x]))
    ax[1, i].imshow(implt.imread(train_path + '/horses/' + train_horses[x]))
fig, ax = plt.subplots(2,4, figsize=(15, 8))
for i in range(4):
    x = random.randint(0, len(test_horses))
    ax[0, i].imshow(implt.imread(test_path + '/humans/' + test_humans[x]))
    ax[1, i].imshow(implt.imread(test_path + '/horses/' + test_horses[x]))
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False
#Commented out model summary because it's output too long. If you wonder uncomment that line and check layers yourself.
#pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
from tensorflow.keras.optimizers import RMSprop

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./ 255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# Validation or test data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1./ 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

validation_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size = 20,
                                                  class_mode = 'binary',
                                                  target_size = (150, 150))
history = model.fit(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 50,
    epochs = 5,
    validation_steps = 12,
    verbose = 2)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.ylim(bottom=0.8)
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()