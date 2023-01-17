# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import imageio
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import subprocess
from IPython import display
import time
from tensorflow import keras 
from skimage import transform
from skimage.color import rgb2gray, gray2rgb

def check_gpu_usage():
    while True:
        display.clear_output(wait=True)
        print(subprocess.check_output('nvidia-smi').decode().strip())
        time.sleep(1)
runner = multiprocessing.Process(target=check_gpu_usage)
runner.start()
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                     ]
        for f in file_names:
            images.append(imageio.imread(f))
            labels.append(d)
    return images, labels
ROOT_PATH = "/kaggle/input/lego-colors"
train_data_dir = os.path.join(ROOT_PATH, "train")
test_data_dir = os.path.join(ROOT_PATH, "test")

train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)
train_images = [transform.resize(image, (500, 500)) for image in train_images]
test_images = [transform.resize(image, (500, 500)) for image in test_images]
unique_labels = set(train_labels)
plt.figure(figsize=(20, 30))
i = 1
for label in unique_labels:
    # Pick the first image for each label.
    image = train_images[train_labels.index(label)]
    plt.subplot(21, 3, i)  # A grid of 8 rows x 8 columns
    plt.axis('off')  
    plt.imshow(image[:,:,0]) # , cmap='gray'
    plt.text(120, 20, "Label {0}".format(label))
    i += 1
plt.show();
def display_label_images(images, label):
    limit = 24  # show a max of 5 images
    plt.figure(figsize=(15, 5))
    i = 1
    start = train_labels.index(label)
    end = start + train_labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)
        plt.axis('off')
        i += 1
        plt.imshow(image[:,:,0])
    plt.show()
    
display_label_images(train_images, "aqua")
train_images = rgb2gray(np.array(train_images))
test_images = rgb2gray(np.array(test_images))
test_labels = np.array(test_labels)
train_labels = np.array(train_labels)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(500, 500)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='softmax')
])
train_labels = pd.factorize(train_labels)[0]
test_labels = pd.factorize(test_labels)[0]
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=30)
test_loss, test_acc = model.evaluate(test_images, test_labels) 
print("Accuracy:", test_acc)     
prediction = model.predict(test_images)
fig = plt.figure(figsize=(26, 20))
for i in range(0, 100, 4):
    plt.subplot(10, 3,1+i/20)
    plt.axis('off')
    plt.text(60, 20, "Truth:    {0}\nPrediction: {1}".format(test_labels[i], np.argmax(prediction[i])), 
             fontsize=12)
    plt.imshow(test_images[i],  cmap="gray")

plt.show()