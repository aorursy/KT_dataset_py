# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import imghdr
import cv2
from os import walk
import re

def main():
    data_dir = pathlib.Path("../input/lego-brick-images/LEGO brick images v1")
    image_count = len(list(data_dir.glob('*.jpg')))
    print(image_count)
    # CLASS NAMES
    class_names = sorted([name for name in os.listdir(data_dir)])
    print(class_names)
    
    batch_size = 128
    img_height = 180
    img_width = 180
        
        
    # CREATE DATASETS
    colormode = "grayscale"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        labels="inferred",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        labels="inferred",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)

    # CREATE TEST DATASET
    test_dir = pathlib.Path("../input/legotest/LegoTest")
    test_classes = []
    for root, dirs, files in os.walk(test_dir):
        #print(root)
        #print(dirs)
        #print(files)
        test_classes.extend(list(map(lambda x: class_names.index(re.sub('\.\.\/input\/legotest\/LegoTest\/', '', root)), files)))
        
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels=test_classes,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)
    
    
        
    print(test_classes[1:])
        
    #CACHE IMAGES
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # CREATE MODEL
    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(img_height, img_width),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    model = tf.keras.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    """
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    """
    epochs = 30
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_ds)

    #SHOW PREDICTIONS
    print(predictions)
    print(test_ds.take(1))
    images, labels = next(iter(test_ds.take(1)))
    labels = labels.numpy()
    print(labels)
    cols = 3
    rows = int((len(predictions)/cols) +2)
    figure = plot_predictions(rows, cols, predictions, labels, images, class_names)
    plot_val_accuracy(rows, cols, hist, figure)
    plt.tight_layout()
    plt.show()

def plot_val_accuracy(num_rows, num_cols, hist, fig):
    max_subplot = num_cols * num_rows * 2
    start = max_subplot - (num_cols*2) +1
    mid = int((max_subplot - start)/2) + start
    print(max_subplot)
    print(start)
    print(mid)
    print(hist.history)
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    train_loss = hist.history['loss']
    plt.subplot(num_rows, num_cols * 2, (start, mid))
    plt.plot(range(len(val_loss)), val_loss, label="validation loss")
    plt.plot(range(len(train_loss)), train_loss, label="training loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(num_rows, num_cols * 2, (mid+1, max_subplot))
    plt.plot(range(len(val_acc)), val_acc, label="max_accuracy-" + str(np.argmax(val_acc)) + "epochs")

    plt.xlabel("epochs")
    plt.ylabel("%")
    plt.legend()

def plot_predictions(num_rows, num_cols, predictions, labels, images, class_names):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = len(predictions)
    #num_images = num_rows * num_cols
    figure = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], labels, images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], labels, len(class_names))
    return figure

def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    #plt.imshow(img.numpy().astype("uint8"))
    #grayscale
    plt.imshow(np.squeeze(img.numpy()),cmap='gray', vmin = 0, vmax = 255)
    #plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% \n({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label, num_classes):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    main()
