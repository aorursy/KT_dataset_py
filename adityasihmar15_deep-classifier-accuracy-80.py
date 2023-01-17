import os

import numpy as np

from tqdm import tqdm

import cv2

import random

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
def load(path):

  imgs = []

  labels = []

  label = -1

  n = 0

  for sbf in os.listdir(path):

    label = label + 1

    n = 0

    pth = os.path.join(path, sbf)

    for ssbf in tqdm(os.listdir(pth)):

      n = n + 1

      img = cv2.imread(os.path.join(pth, ssbf),0)

      img = cv2.resize(img, img_size)

      imgs.append((img/255).reshape(img_size[0],img_size[1],1))

      labels.append(label)



  tmp = list(zip(imgs, labels))

  random.shuffle(tmp)

  imgs, labels = zip(*tmp)

  return np.array(imgs), np.array(labels)

def get_classlabel(indx):

    labels = {0:'buildings',1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

    return labels[indx]
path_train = '/kaggle/input/intel-image-classification/seg_train/seg_train'

path_test = '/kaggle/input/intel-image-classification/seg_test/seg_test'

path_pred = '/kaggle/input/intel-image-classification/seg_pred/seg_pred'

img_size = (120,120)
imgs, labels = load(path_train)
print(labels[0])
import tensorflow as tf

import keras

import pprint
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

strategy = tf.distribute.experimental.TPUStrategy(tpu)
with strategy.scope():

    seq = keras.Sequential()

    seq.add(tf.keras.layers.Conv2D(64, (2,2), activation='relu'))

    seq.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))

    seq.add(tf.keras.layers.MaxPool2D((3,3)))

    #seq.add(tf.keras.layers.BatchNormalization())

    seq.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))

    seq.add(tf.keras.layers.Conv2D(256, (2,2), activation='relu'))

    seq.add(tf.keras.layers.MaxPool2D((3,3)))

    #seq.add(tf.keras.layers.BatchNormalization())

    seq.add(tf.keras.layers.Flatten())

    seq.add(tf.keras.layers.Dense(50, activation='relu'))

    seq.add(tf.keras.layers.Dropout(0.5))

    seq.add(tf.keras.layers.Dense(30, activation='relu'))

    seq.add(tf.keras.layers.Dropout(0.4))

    seq.add(tf.keras.layers.Dense(6, activation='softmax'))

    

    seq.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), metrics='accuracy')
history = seq.fit(imgs, labels, batch_size=8*128, epochs=100, shuffle=True, validation_split=0.2)
%matplotlib inline

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
tst, tst_lbl = load(path_test)
result = seq.evaluate(tst, tst_lbl)

print('Test Accuracy = ',  result[1])