# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.layers import Input, Dense

from keras.models import Model

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.noise import GaussianNoise

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.utils import np_utils

from keras.callbacks import History

from keras import regularizers

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split



import pandas as pd 

import numpy as np



columns = [

    "title",

    "year",

    "lifetime_gross",

    "ratingInteger",

    "ratingCount",

    "duration",

    "nrOfWins",

    "nrOfNominations",

    "nrOfPhotos",

    "nrOfNewsArticles",

    "nrOfUserReviews",

    "nrOfGenre",

    "Action",

    "Adult" ,

    "Adventure",

    "Animation",

    "Biography",

    "Comedy",

    "Crime",

    "Documentary",

    "Drama",

    "Family",

    "Fantasy",

    "Horror",

    "Music",

    "Musical",

    "Mystery",

    "News",

    "RealityTV",

    "Romance",

    "SciFi",

    "Short",

    "Sport",

    "TalkShow",

    "Thriller",

    "War",

    "Western"]

dt = pd.read_csv('../input/movies-example-for-machine-learning-activities/MACHINE_LEARNING_FINAL.csv', header=None, names=columns, sep=";")



def plot_history(network_history):

    plt.figure(figsize=(14,7))

    ax = plt.subplot(1, 2, 1)

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.plot(x_plot, network_history.history['loss'])

    plt.plot(x_plot, network_history.history['val_loss'])

    plt.legend(['Training', 'Validation'])



    ax = plt.subplot(1, 2, 2)

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.plot(x_plot, network_history.history['accuracy'])

    plt.plot(x_plot, network_history.history['val_accuracy'])

    plt.legend(['Training', 'Validation'], loc='lower right')



    plt.show()



def plot_representation(x_test,encoded_imgs,codex,codey,decoded_imgs):

    n = 14

    plt.figure(figsize=(16, 3))

    for i in range(n):

        # display original

        ax = plt.subplot(3, n, i + 1)

        plt.imshow(x_test[i].reshape(28, 28))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)



        ax = plt.subplot(3, n, i + 1 + n)

        plt.imshow(encoded_imgs[i].reshape(codex, codey))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)



        # display reconstruction

        ax = plt.subplot(3, n, i + 1 + 2*n)

        plt.imshow(decoded_imgs[i].reshape(28, 28))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)



    plt.show()

    

def preprocess_labels(labels, encoder=None, categorical=True):

    if not encoder:

        encoder = LabelEncoder()

        encoder.fit(labels)

    y = encoder.transform(labels).astype(np.int32)

    if categorical:

        y = np_utils.to_categorical(y)

    return y, encoder
dt.shape
dt.columns
dati = dt.loc[1:, dt.columns != 'ratingInteger']

dati = dati.loc[:, dati.columns != 'title']

Y = dt['ratingInteger']

Y = Y[1:]
dati.shape
Y.shape
train , test , labels , tlabels = train_test_split(dati, Y)

print(train.shape)

print(test.shape)
labels.value_counts()
tlabels.value_counts()
labels, encoder = preprocess_labels(labels)

labels
tlabels, tencoder = preprocess_labels(tlabels)

tlabels
n_epochs = 100

b = 100

dims=len(train.columns)

nb_classes=len(Y.unique())



x_plot = list(range(1,n_epochs+1))



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(dims,)))

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network_history = model.fit(train, labels, epochs=n_epochs, batch_size=b, shuffle=True, validation_data=(test, tlabels))

model.summary()
plot_history(network_history)



score = model.evaluate(test,tlabels, batch_size=b) #evaluating the models accuracy or loss,

print('test loss, test acc:', score)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print("\n%s: %.2f" % (model.metrics_names[0], score[0]))
from sklearn import preprocessing



x = dati.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

data = pd.DataFrame(x_scaled) # <-- Look! this is another variable



seed = 2

np.random.seed(seed)
train , test , labels , tlabels = train_test_split(data, Y)

print(train.shape)

print(test.shape)

print(labels.value_counts())

print(tlabels.value_counts())



labels, encoder = preprocess_labels(labels)

print(labels)



tlabels, tencoder = preprocess_labels(tlabels)

print(tlabels)

n_epochs = 200

b = 100

dims=len(train.columns)

nb_classes=len(Y.unique())



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(dims,)))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network_history = model.fit(train, labels, epochs=n_epochs, batch_size=b, shuffle=True, validation_data=(test, tlabels))

model.summary()
x_plot = list(range(1,n_epochs+1))



plot_history(network_history)



score = model.evaluate(test,tlabels, batch_size=b) #evaluating the models accuracy or loss,

print('test loss, test acc:', score)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print("\n%s: %.2f" % (model.metrics_names[0], score[0]))
n_epochs = 200

b = 200

dims=len(train.columns)

nb_classes=len(Y.unique())



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(dims,)))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network_history = model.fit(train, labels, epochs=n_epochs, batch_size=b, shuffle=True, validation_data=(test, tlabels))

model.summary()
x_plot = list(range(1,n_epochs+1))



plot_history(network_history)



score = model.evaluate(test,tlabels, batch_size=b) #evaluating the models accuracy or loss,

print('test loss, test acc:', score)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print("\n%s: %.2f" % (model.metrics_names[0], score[0]))
n_epochs = 200

b = 512

dims=len(train.columns)

nb_classes=len(Y.unique())



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(dims,)))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network_history = model.fit(train, labels, epochs=n_epochs, batch_size=b, shuffle=True, validation_data=(test, tlabels))

model.summary()
x_plot = list(range(1,n_epochs+1))



plot_history(network_history)



score = model.evaluate(test,tlabels, batch_size=b) #evaluating the models accuracy or loss,

print('test loss, test acc:', score)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print("\n%s: %.2f" % (model.metrics_names[0], score[0]))
n_epochs = 200

b = 512

dims=len(train.columns)

nb_classes=len(Y.unique())



model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(dims,)))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network_history = model.fit(train, labels, epochs=n_epochs, batch_size=b, shuffle=True, validation_data=(test, tlabels))

model.summary()



x_plot = list(range(1,n_epochs+1))



plot_history(network_history)



score = model.evaluate(test,tlabels, batch_size=b) #evaluating the models accuracy or loss,

print('test loss, test acc:', score)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print("\n%s: %.2f" % (model.metrics_names[0], score[0]))