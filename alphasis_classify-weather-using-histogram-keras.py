import numpy as np

import pandas as pd

import pickle

import skimage

from skimage import io, transform, exposure

from matplotlib import pyplot as plt



MAX_NUM = 20000
csv = pd.read_csv(r'../input/train_v2.csv')

ser = csv['tags'].map(lambda x : x.split())

tags = []

for s in ser:

    tags = tags + s

ser = pd.Series(tags)

ser.value_counts()
sel_tags = ['clear', 'haze', 'cloudy', 'partly_cloudy']

def is_within_labels(x):

    x = x.split()

    for i in x:

        if i in sel_tags:

            return True

    return False

sel_rows = csv['tags'].map(is_within_labels)

np.sum(sel_rows)
sel_csv = csv.loc[sel_rows]

def trans_tags(x):

    x = x.split()

    y = []

    for i in x:

        if i in sel_tags:

            y.append(i)

    return " ".join(y)

sel_csv['tags'] = sel_csv['tags'].map(trans_tags)

sel_csv = sel_csv.drop(sel_csv.index[MAX_NUM:])

sel_csv.to_csv(r'weather_sel.csv', index=False)
#transforming tags

counts = 0

tags_df = pd.DataFrame(columns=sel_tags)

for index in sel_csv.index:

    row = sel_csv.ix[index]

    new_row = []

    for i in sel_tags:    

        if i in row['tags']:

            new_row.append(int(1))

        else: new_row.append(int(0))

    tags_df.loc[index] = new_row

    counts = counts + 1

    if(counts == MAX_NUM): break

tags_df.head()
#scaling images

from skimage import io

import skimage

from skimage.transform import *

import matplotlib.pyplot as plt

import os
ims = []

base_path = r'../input/train-jpg'

coutns = 0

for index in sel_csv.index:

    im_path = os.path.join(base_path, sel_csv.ix[index]['image_name'] + '.jpg')

    ti = io.imread(im_path)

    ti = resize(ti, (50, 50), mode='edge')

    ims.append(ti)

    counts = counts + 1

    if(counts == MAX_NUM): break

ims = np.array(ims)

tags_arr = tags_df.values[:MAX_NUM]

print(ims.shape)

print(tags_arr.shape)
#we select train_2 and train_88 to illustrate histogram usages

ti1 = io.imread(r'../input/train-jpg/train_2.jpg')

ti2 = io.imread(r'../input/train-jpg/train_88.jpg')

ti1 = transform.resize(ti1, (50, 50), mode='edge')

ti2 = transform.resize(ti2, (50, 50), mode='edge')



plt.subplot(1,2,1)

plt.imshow(ti1)

plt.subplot(1,2,2)

plt.imshow(ti2)

plt.show()
NUM_OF_BINS = 50

ti1_hist = exposure.histogram(ti1, nbins=NUM_OF_BINS)

ti2_hist = exposure.histogram(ti2, nbins=NUM_OF_BINS)



xaxis = np.arange(NUM_OF_BINS)

plt.bar(xaxis, +ti1_hist[0], facecolor='#9999ff', edgecolor='white')

plt.bar(xaxis, -ti2_hist[0], facecolor='#ff9999', edgecolor='white')

#plt.ylim(-200, +200)

plt.show()
ims_hist = []

for im in ims:

    ims_hist.append(exposure.histogram(im, nbins=NUM_OF_BINS)[0])

ims_hist = np.array(ims_hist)
#Finally, build the neural network

import numpy as np

import tensorflow as tf

import sklearn

from keras.models import Sequential, Model

from keras.layers import Dense, Input, Activation, Flatten, Lambda, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU

from keras import optimizers, regularizers, initializers, metrics, activations, losses

import keras

import pandas as pd



config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)



data = ims_hist

target = tags_arr



print(data.shape)

print(target.shape)



seed = 3

np.random.seed(seed=seed)

batch_size = 10



def threshold_binarize(x, threshold):

    ge = tf.greater_equal(x, tf.constant(threshold))

    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))

    return y



def threshold_metrics(y_true, y_pred):

    y_pred = threshold_binarize(y_pred, threshold=0.25)

    return metrics.binary_accuracy(y_true, y_pred)



def pre_train(X_train, hidden_layers,

              decoder_activation='sigmoid',

              batch_size=100, pre_train_epoch=1):



    for i in np.arange(len(hidden_layers)):

        print('Pre-training the layer: Input {} -> Output {}'.format(hidden_layers[i].input_shape, hidden_layers[i].output_shape))



        # Create AE and training

        encoded = hidden_layers[i]

        decoded = Dense(encoded.input_shape[1], activation=decoder_activation)

        ae_model = Sequential()

        ae_model.add(encoded)

        ae_model.add(decoded)



        ae_model.compile(loss='mse', optimizer='rmsprop')

        if i == 0:

            # Train the simple autoencoder

            ae_model.fit(X_train, X_train,

                         batch_size=batch_size, nb_epoch=pre_train_epoch, verbose=False)

        else:

            pre_model = Sequential()

            for i in range(i):

                pre_model.add(hidden_layers[i])



            X_data = pre_model.predict(X_train)

            ae_model.fit(X_train, X_data,

                         batch_size=batch_size, nb_epoch=pre_train_epoch, verbose=False)



input_shape = data.shape[1:]

layers = [Dense(1000, activation='tanh',

                input_shape=input_shape,

                kernel_initializer=initializers.RandomNormal(stddev=10)),

          Dense(100, activation='tanh',

                kernel_initializer=initializers.RandomNormal(stddev=10)),

          Dense(4, activation='sigmoid',

                kernel_initializer=initializers.RandomNormal(stddev=10))]



model = Sequential()

for layer in layers:

    model.add(layer)



pre_train(data, layers)



opt = optimizers.Adam(lr=0.01)

model.compile(loss=losses.binary_crossentropy, optimizer=opt, metrics=[threshold_metrics])



#Fit model on training data

model.fit(data, target, batch_size=batch_size, nb_epoch=1, verbose=True)



result = model.predict(x=data[400:], batch_size=batch_size, verbose=True)



predict = pd.DataFrame(data=result, columns=['clear', 'haze', 'cloudy', 'partly_cloudy'])

def binarize(row):

    row[row >= 0.25] = 1

    row[row < 0.25] = 0

    return row

for index in predict.index:

    predict.ix[index] = binarize(predict.ix[index])

predict.head()



predict.to_csv(r'predict.csv', index=False)