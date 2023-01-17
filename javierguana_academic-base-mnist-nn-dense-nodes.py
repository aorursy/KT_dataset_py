# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 

import tensorflow as tf

from matplotlib import pyplot

import matplotlib as mpl



%matplotlib inline





# Any results you write to the current directory are saved as output.



from keras import models

from keras import layers

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_file = '../input/fashion-mnist_train.csv'

test_file = '../input/fashion-mnist_test.csv'
def read_dataset(data_file):

    df = pd.read_csv(data_file)

    label_column = 'label'

    y = df[label_column].values

    X = df.drop(label_column, axis=1).values

    return (X, y)
(train_X, train_y) = read_dataset(train_file)

(test_X, test_y) = read_dataset(test_file)
def draw_articles(articles, labels):

    fig, axs = plt.subplots(1, len(articles), figsize=(30,30))

    for i in range(len(articles)):

        axs[i].set_title(labels[i])

        axs[i].imshow(articles[i].reshape((28,28)), cmap=plt.cm.binary)

    plt.show()
def ImageDisplay(list_data, label, one_hot=False):

    fig = pyplot.figure()

    axis = fig.add_subplot(1,1,1)

    list_data=np.reshape(list_data, (28,28))

    plot_img = axis.imshow(list_data, cmap=mpl.cm.Greys)

    plot_img.set_interpolation('none')

    if one_hot :

        ShowLabelName (label)

    else:

        print ("Label : "+str(CLASSES[str(label)]))
label_map = {0: 'T-Shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

examples = []

labels = []



for i in label_map:

    k = np.where(train_y==i)[0][0]

    examples.append(train_X[k])

    labels.append(label_map[i])

draw_articles(examples, labels)
train_X = train_X.astype('float32') / 255

train_y = to_categorical(train_y)



test_X = test_X.astype('float32') / 255

test_y = to_categorical(test_y)
val_X = train_X[:10000]

train_X = train_X[10000:]



val_y = train_y[:10000]

train_y = train_y[10000:]
network = models.Sequential()

network.add(layers.Dense(256, activation='relu', input_shape=(784,)))

network.add(layers.Dense(128, activation='relu', input_shape=(784,)))

network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',

               loss='categorical_crossentropy',

               metrics=['accuracy'])
network.fit(train_X, train_y, epochs=10, batch_size=128, validation_data=(val_X, val_y))
test_loss, test_acc = network.evaluate(test_X, test_y)

print('test_loss=', test_loss)

print('test_accuracy=', test_acc)