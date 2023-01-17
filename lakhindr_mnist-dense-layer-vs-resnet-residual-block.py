import os
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, Input, Add
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
def tf_tutorial_orig_dense_model(dropout=0.4, dense_size=1024):
    """See https://www.tensorflow.org/tutorials/layers for the basic architecture implemented below.
       First two are (5,5) Convolution layers -- each followd by a maxpooling of (2,2) layers.
       The next layer is a Dense layer of 1024 (default) neurons: a total of over 3 Million parameters """
    img = Input(shape=(1,28,28))
    # first convolutional layer:
    c1 = Convolution2D(32, (5,5), activation='relu', padding='same', data_format='channels_first') (img)
    m1 = MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first')(c1)
    # second convolutional layer:
    c2 = Convolution2D(64, (5,5), activation='relu', padding='same', data_format='channels_first') (m1)
    m2 = MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first')(c2)
    f2 = Flatten()(m2)
    # first dense layer:
    d3 = Dense(dense_size, activation='relu')(f2)
    if dropout > 0:
        d3 = Dropout(dropout)(d3)
    # final (dense) layer:
    smax4 = Dense(10, activation='softmax')(d3)
    
    sgd = SGD(lr=0.001)
    model = Model(inputs=img, outputs=smax4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def tf_tutorial_resnet_model():
    """ This model keeps the first two Convolutional (5,5) layers, as the other model below, but replaces 
        its Dense layer by Residual groups, as described in ResNet paper: """
    img = Input(shape=(1,28,28))
    # first 2 convolutional layers (same as the other model):
    c1 = Convolution2D(32, (5,5), activation='relu', padding='same', data_format='channels_first') (img)
    m1 = MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first')(c1)
    c2 = Convolution2D(64, (5,5), activation='relu', padding='same', data_format='channels_first') (m1)
    m2 = MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first')(c2)
    
    # no dense layer, or dropout is used. It is replaced with a resnet style block:
    a5 = resnet_conv_blk(m2)
    
    # final (dense/softmax) layer (same as the other model):
    f = Flatten()(a5)
    smax4 = Dense(10, activation='softmax')(f)
    
    sgd = SGD(lr=0.001)
    model = Model(inputs=img, outputs=smax4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def resnet_conv_blk(inp):
    """This is a modified ResNet block: a set of two convolutional layers of [3x3,64] filters each
       The first layer output is not Batch Normalized, as it feeds directly into the second Conv. layer """
    c1 = Convolution2D(64, (3,3), activation='relu', use_bias=False, padding='same', data_format='channels_first') (inp)
    #c1 = BatchNormalization(epsilon=1.1e-5, axis=1)(c1)  # This is not used, unlike in ResNet.
    c2 = Convolution2D(64, (3,3), activation=None, padding='same', data_format='channels_first') (c1)
    c2 = BatchNormalization(epsilon=1.1e-5, axis=1)(c2) # axis =1 for 'channels_first' mode.
    inp = BatchNormalization(epsilon=1.1e-5, axis=1)(inp)
    a3 = Add()([inp, c2])
    out = Activation('relu')(a3)
    return out
# little plotter helper to see results. various dense layer data is plotted simultaneously against the epochs.
def plotxy(hist, title="tf-mnist", ytitle="accuracy", xtitle="epoch", legend=()):
    fig = plt.figure() #change dpi to get bigger resolution..
    for h in hist:
        plt.plot(h)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.legend(legend, loc='lower right')
    plt.title(title)
    plt.show()
    plt.close(fig)
# this is just to pull data off the history object. Look at keras documentation for history. It gives back data in columns, and their labels.
def get_data_n_legend(hist, id):
    plot_data = []
    legend = []
    for d in sorted(hist.keys()):
        plot_data.append(hist[d].history[id])
        legend.append(str(d))  # legends shown in the plot. E.g. d-128, d-256 denote the dense layer size.
    return((plot_data, legend))
        
# make a graph of the history gathered..
def print_hist(hist, title="tf-mnist dense layers", id = "val_acc"):
    (pl_data, legnd) = get_data_n_legend(hist, id)
    plotxy(pl_data, title, xtitle='epoch', legend=legnd, ytitle=id)
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
x_train = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(train_data['label'])
hist_d = {}
for dense_units in (128, 1024):
    modl_d = tf_tutorial_orig_dense_model(dense_size=dense_units)
    print("Training Dense Model for dense layer of size:", dense_units)
    hist_d['d-' + str(dense_units)] = modl_d.fit(x_train, y_train, epochs=16, batch_size=32, validation_split=0.5)
hist_rb = {}
modl_rb= tf_tutorial_resnet_model()
print("Training ResNet block based Model")
hist_rb['ResNet-Blk-1'] = modl_rb.fit(x_train, y_train, epochs=16, batch_size=32, validation_split=0.5)
print_hist(hist_d, title="tf-mnist Dense Model & dense layer size")
print_hist(hist_rb, title="tf-mnist ResNet block based Model")
modl_d.summary()
modl_rb.summary()