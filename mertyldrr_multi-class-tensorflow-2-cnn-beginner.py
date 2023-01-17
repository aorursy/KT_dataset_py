import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as implt
import os
train_dir = "/kaggle/input/rock-paper-scissor/rps/rps"
test_dir = "/kaggle/input/rock-paper-scissor/rps-test-set/rps-test-set"

train_rock = os.listdir(train_dir + "/rock")
train_paper = os.listdir(train_dir + "/paper")
train_scissors = os.listdir(train_dir + "/scissors")

test_rock = os.listdir(test_dir + "/rock")
test_paper = os.listdir(test_dir + "/paper")
test_scissors = os.listdir(test_dir + "/scissors")
print("Number of images in the train-set:", len(train_rock) + len(train_paper) + len(train_scissors))
print("Number of images in the test-set:", len(test_rock) + len(test_paper) + len(test_scissors))

print("\nNumber of rocks in the train-set:", len(train_rock))
print("Number of papers in the train-set:", len(train_paper))
print("Number of scissors in the train-set:", len(train_scissors))

print("\nNumber of rocks in the test-set:", len(test_rock))
print("Number of papers in the test-set:", len(test_paper))
print("Number of scissors in the test-set:", len(test_scissors))
import random

fig, ax = plt.subplots(3,4, figsize=(12, 8))
for i in range(4):
    x = random.randint(0, len(train_rock))
    #Rock
    ax[0, i].imshow(implt.imread(train_dir + '/rock/' + train_rock[x]))
    ax[0, 0].set_ylabel('rock')
    #Paper
    ax[1, i].imshow(implt.imread(train_dir + '/paper/' + train_paper[x]))
    ax[1, 0].set_ylabel('paper')
    #Scissors
    ax[2, i].imshow(implt.imread(train_dir + '/scissors/' + train_scissors[x]))
    ax[2, 0].set_ylabel('scissors')
fig, ax = plt.subplots(3,4, figsize=(12, 8))
for i in range(4):
    x = random.randint(0, len(test_rock))
    #Rock
    ax[0, i].imshow(implt.imread(test_dir + '/rock/' + test_rock[x]))
    ax[0, 0].set_ylabel('rock')
    #Paper
    ax[1, i].imshow(implt.imread(test_dir + '/paper/' + test_paper[x]))
    ax[1, 0].set_ylabel('paper')
    #Scissors
    ax[2, i].imshow(implt.imread(test_dir + '/scissors/' + test_scissors[x]))
    ax[2, 0].set_ylabel('scissors')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150, 150),
                                                    class_mode = 'categorical',
                                                    batch_size = 126)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                 target_size = (150, 150),
                                                 class_mode = 'categorical',
                                                 batch_size = 126)
my_callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
my_callback_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=0.00001, verbose=1)

model = tf.keras.models.Sequential([
    # Conv layer-1
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    # Conv layer-2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    # Conv layer-3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    # Conv layer-4
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    # Flatten output from Conv-4 
    tf.keras.layers.Flatten(),
    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    # We have three class
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',              
              optimizer='rmsprop',
              metrics = ['accuracy'])
history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs=25,
                    steps_per_epoch= 20, #(2520 // 126), # train-set size = 2520, batch_size = 126
                    validation_steps= 3, #(372 // 126), # train-set size = 372, batch_size = 126
                    verbose = 1,
                    callbacks=[my_callback_es, my_callback_rlr])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.ylim(bottom=0.8)
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()