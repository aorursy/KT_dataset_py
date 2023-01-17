import tensorflow as tf



from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt



import os

from os.path import join

import numpy as np
#loaddata

#from six.moves import cPickle as pickle

import pickle as p

import platform

img_rows, img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)

def load_pickle(f):

    version = platform.python_version_tuple()

    if version[0] == '2':

        return  pickle.load(f)

    elif version[0] == '3':

        return  pickle.load(f, encoding='latin1')

    raise ValueError("invalid python version: {}".format(version))

    

def load_CIFAR_batch(filename):

    """ load single batch of cifar """

    with open(filename, 'rb') as f:

        #datadict = load_pickle(f)

        datadict = p.load(f, encoding='latin1')

        X = datadict['data']

        Y = datadict['labels']

        #X = X.reshape(10000,3072)

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

        Y = np.array(Y)

        return X, Y

def load_CIFAR10(ROOT):

    """ load all of cifar """

    xs = []

    ys = []

    for b in range(1,6):

        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))

        X, Y = load_CIFAR_batch(f)

        xs.append(X)

        ys.append(Y)    

  

    Xtr = np.concatenate(xs)

    Ytr = np.concatenate(ys)

    del X, Y

    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    return Xtr, Ytr, Xte, Yte
image_dir = '../input/cifar-10-batches-py/'

train_images, train_labels, test_images, test_labels = load_CIFAR10(image_dir)



# Normalize pixel values to be between 0 and 1

train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    # The CIFAR labels happen to be arrays, 

    # which is why you need the extra index

    plt.xlabel(class_names[train_labels[i]])

plt.show()
inputs = layers.Input(shape=(32, 32, 3))

x = layers.Conv2D(32,(3, 3), activation='relu')(inputs)

x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64,(3, 3), activation='relu')(x)

x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64,(3, 3), activation='relu')(x)

y = layers.Flatten()(x)

y = layers.Dense(64, activation='relu')(y)

predictions = layers.Dense(10, activation='softmax')(y)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



history = model.fit(train_images, train_labels, epochs=10, 

                    validation_data=(test_images, test_labels))
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
model.save_weights('trained_weights_final.h5')