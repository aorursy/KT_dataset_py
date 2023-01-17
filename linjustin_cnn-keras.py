# Import necessary libraries
import os
import pathlib
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, Flatten, Dropout,
                                     MaxPooling2D, Activation, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('TensorFlow Version: ', tf.__version__)
print('GPU Available: ', tf.test.is_gpu_available())
print('Using GPU: ', tf.config.experimental.list_physical_devices('GPU'))
PATH = '../input/train-val-test-tcga-coad-msi-mss/tcga_coad_msi_mss/'
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

train_msimut_dir = os.path.join(train_dir, 'MSIMUT')
train_mss_dir = os.path.join(train_dir, 'MSS')
val_msimut_dir = os.path.join(val_dir, 'MSIMUT')
val_mss_dir = os.path.join(val_dir, 'MSS')
test_msimut_dir = os.path.join(test_dir, 'MSIMUT')
test_mss_dir = os.path.join(test_dir, 'MSS')
# Check how many images are in each directory
num_msimut_train, num_mss_train = len(os.listdir(train_msimut_dir)), len(os.listdir(train_mss_dir))

num_msimut_val, num_mss_val = len(os.listdir(val_msimut_dir)), len(os.listdir(val_mss_dir))

num_msimut_test, num_mss_test = len(os.listdir(test_msimut_dir)), len(os.listdir(test_mss_dir))

total_train = num_msimut_train + num_mss_train
total_val = num_msimut_val + num_mss_val
total_test = num_msimut_test + num_mss_test

print('Total training MSIMUT images: ', num_msimut_train)
print('Total training MSS images: ', num_mss_train)
print('Total validation MSIMUT images: ', num_msimut_val)
print('Total validation MSS images: ', num_mss_val)
print('Total testing MSIMUT images: ', num_msimut_test)
print('Total testing MSS images: ', num_mss_test)
print('---------------------------------')
print('Total training images: ', total_train)
print('Total validation images: ', total_val)
print('Total testing images: ', total_test)
# Set up variables for pre-processing
batch_size = 64
epochs = 5
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Visualize some images
train_root = pathlib.Path(train_dir)
class_names = sorted([j.name.split('/')[-1] for j in train_root.iterdir()])
class_names = np.array(class_names)
print('Class names: ', class_names)

img_gen = ImageDataGenerator(rescale = 1./255)
sample_train_data_gen = img_gen.flow_from_directory(batch_size = batch_size,
                                                    directory = train_dir,
                                                    shuffle = True,
                                                    target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                    classes = list(class_names))
                                                 
sample_images, sample_labels = next(sample_train_data_gen)

def show_batch(img_batch, label_batch):
    plt.figure(figsize = (10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(sample_images[i])
        plt.title(class_names[sample_labels[i] == 1][0])
        plt.axis('off')
        
show_batch(sample_images, sample_labels)
train_image_generator = ImageDataGenerator(rescale = 1./255,
                                           rotation_range = 45,
                                           width_shift_range = 0.20,
                                           height_shift_range = 0.20,
                                           horizontal_flip = True,
                                           zoom_range = 0.5)

val_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size = batch_size,
                                                           directory = train_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode = 'binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size = batch_size,
                                                       directory = val_dir,
                                                       target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode = 'binary')
# Create CNN
model = Sequential([
    # Conv layer 1/Input layer
    Conv2D(64, kernel_size = (5, 5),padding = 'same', activation = 'relu', 
           input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.25),
    
    # Conv layer 2
    Conv2D(64, kernel_size = (5, 5), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.25),

    # Conv layer 3
    Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
    Dropout(0.25),
    
    # Conv layer 4
    Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
    Dropout(0.25),

    # Fully connected layer 1
    Flatten(),
    Dense(256, activation = 'relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    # Fully connected last layer
    Dense(1, activation = 'sigmoid')
])

# Standard metrics for binary classification 
metrics = [
    tf.keras.metrics.TruePositives(name = 'tp'),
    tf.keras.metrics.FalsePositives(name = 'fp'),
    tf.keras.metrics.TrueNegatives(name = 'tn'),
    tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
    tf.keras.metrics.Precision(name = 'precision'),
    tf.keras.metrics.Recall(name = 'recall'),
    tf.keras.metrics.AUC(name = 'auc')
]

initial_lr = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps = 100000,
    decay_rate = 0.96,
    staircase = True
)

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule,
                                                  momentum = 0.9,
                                                  nesterov = True),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = metrics)

model.summary()
print('Starting training...')
print('====================\n')

start = time()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1e-2,
        patience = 2,
        verbose = 1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = '../output/cnn.ckpt',
        save_best_only = True,
        monitor = 'val_loss',
        verbose = 0
    )
]

history = model.fit(train_data_gen,
                    steps_per_epoch = total_train // batch_size,
                    epochs = epochs,
                    validation_data = val_data_gen,
                    validation_steps = total_val // batch_size,
                    callbacks = callbacks)

end = time()

time_elapsed = end - start

print('\nTraining took {:.0f}h {:.0f}m {:.0f}s.'.format(time_elapsed//(60*60),
                                                        time_elapsed//60, 
                                                        time_elapsed % 60))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()
test_image_generator = ImageDataGenerator(rescale = 1./255)
test_data_gen = test_image_generator.flow_from_directory(batch_size = batch_size,
                                                         directory = test_dir,
                                                         shuffle = False,
                                                         target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode = 'binary')

result = model.evaluate(test_data_gen)
print('Test Loss: ', result[0])
print('Test Accuracy: ', result[4])
print('Test AUC: ', result[7])
test_image, test_label = next(test_data_gen)

predicted_batch = model.predict(test_image)
predicted_id = np.argmax(predicted_batch, axis = -1)
predicted_label_batch = class_names[predicted_id]

plt.figure(figsize = (10, 10))
plt.subplots_adjust(hspace = 0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(test_image[n])
    color = "blue" if predicted_id[n] == test_label[n] else "red"
    plt.title(predicted_label_batch[n], color = color)
    plt.axis('off')
_ = plt.suptitle("CNN Predictions (blue: correct, red: incorrect)")