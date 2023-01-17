import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import os

import wave

import pylab

from pathlib import Path

from scipy import signal

from scipy.io import wavfile

from sklearn.metrics import confusion_matrix

import itertools



# Set paths to input and output data

INPUT_DIR = '/kaggle/input/free-spoken-digits/free-spoken-digit-dataset-master/recordings/'

OUTPUT_DIR = '/kaggle/working/'



# Print names of 10 WAV files from the input path

parent_list = os.listdir(INPUT_DIR)

for i in range(10):

    print(parent_list[i])
# Plot first 5 WAV files as a waveform and a frequency spectrum

for i in range(5): 

    signal_wave = wave.open(os.path.join(INPUT_DIR, parent_list[i]), 'r')

    sample_rate = 16000

    sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)



    plt.figure(figsize=(12,12))

    plot_a = plt.subplot(211)

    plot_a.set_title(parent_list[i])

    plot_a.plot(sig)

    plot_a.set_xlabel('sample rate * time')

    plot_a.set_ylabel('energy')



    plot_b = plt.subplot(212)

    plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)

    plot_b.set_xlabel('Time')

    plot_b.set_ylabel('Frequency')



plt.show()
# Utility function to get sound and frame rate info

def get_wav_info(wav_file):

    wav = wave.open(wav_file, 'r')

    frames = wav.readframes(-1)

    sound_info = pylab.frombuffer(frames, 'int16')

    frame_rate = wav.getframerate()

    wav.close()

    return sound_info, frame_rate



# For every recording, make a spectogram and save it as label_speaker_no.png

if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):

    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))

    

for filename in os.listdir(INPUT_DIR):

    if "wav" in filename:

        file_path = os.path.join(INPUT_DIR, filename)

        file_stem = Path(file_path).stem

        target_dir = f'class_{file_stem[0]}'

        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)

        file_dist_path = os.path.join(dist_dir, file_stem)

        if not os.path.exists(file_dist_path + '.png'):

            if not os.path.exists(dist_dir):

                os.mkdir(dist_dir)

            file_stem = Path(file_path).stem

            sound_info, frame_rate = get_wav_info(file_path)

            pylab.specgram(sound_info, Fs=frame_rate)

            pylab.savefig(f'{file_dist_path}.png')

            pylab.close()



# Print the ten classes in our dataset

path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images'))

print("Classes: \n")

for i in range(10):

    print(path_list[i])

    

# File names for class 1

path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images/class_1'))

print("\nA few example files: \n")

for i in range(10):

    print(path_list[i])
# Declare constants

IMAGE_HEIGHT = 256

IMAGE_WIDTH = 256

BATCH_SIZE = 32

N_CHANNELS = 3

N_CLASSES = 10



# Make a dataset containing the training spectrograms

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(

                                             batch_size=BATCH_SIZE,

                                             validation_split=0.2,

                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),

                                             shuffle=True,

                                             color_mode='rgb',

                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

                                             subset="training",

                                             seed=0)



# Make a dataset containing the validation spectrogram

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(

                                             batch_size=BATCH_SIZE,

                                             validation_split=0.2,

                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),

                                             shuffle=True,

                                             color_mode='rgb',

                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

                                             subset="validation",

                                             seed=0)
plt.figure(figsize=(12, 12))

for images, labels in train_dataset.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")

plt.show()
# Function to prepare our datasets for modelling

def prepare(ds, augment=False):

    # Define our one transformation

    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])

    flip_and_rotate = tf.keras.Sequential([

        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),

        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)

    ])

    

    # Apply rescale to both datasets and augmentation only to training

    ds = ds.map(lambda x, y: (rescale(x, training=True), y))

    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))

    return ds



train_dataset = prepare(train_dataset, augment=False)

valid_dataset = prepare(valid_dataset, augment=False)
# Create CNN model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))

model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))



# Compile model

model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer=tf.keras.optimizers.RMSprop(),

    metrics=['accuracy'],

)



# Train model for 10 epochs, capture the history

history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
# Plot the loss curves for training and validation.

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)



plt.figure(figsize=(8,6))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Plot the accuracy curves for training and validation.

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(acc_values)+1)



plt.figure(figsize=(8,6))

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Compute the final loss and accuracy

final_loss, final_acc = model.evaluate(valid_dataset, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
# Clean the output dir

import shutil

shutil.rmtree('/kaggle/working/audio-images')