import os
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, Input
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
def tf_tutorial_keras_sequential_model(dropout=0.4, dense_size=1024):
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(32, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(64, (5,5), padding='same', activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', data_format='channels_first'))
    # first dense layer:
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    # final (dense) layer:
    model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def tf_tutorial_keras_func_model(dropout=0.4, dense_size=1024):
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
        d3 = Dropout(dropout, name='Dropout')(d3)
    # final (dense) layer:
    smax4 = Dense(10, activation='softmax')(d3)
    
    sgd = SGD(lr=0.001)
    model = Model(inputs=img, outputs=smax4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
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
        legend.append('d-' + str(d))  # legends e.g. d-128, d-256 denote the dense layer size.
    return((plot_data, legend))
        
# make a graph of the history gathered..
def print_hist(hist, title="tf-mnist dense layers", id = "val_acc"):
    (pl_data, legnd) = get_data_n_legend(hist, id)
    plotxy(pl_data, title, xtitle='epoch', legend=legnd, ytitle=id)
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
x_train = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(train_data['label'])
hist_seq = {}
for dense_units in (128, 1024):
    modl_seq = tf_tutorial_keras_sequential_model(dense_size=dense_units)
    print("Training Keras Sequential Model for dense layer of size:", dense_units)
    hist_seq[dense_units] = modl_seq.fit(x_train, y_train, epochs=16, batch_size=32, validation_split=0.5)
hist_func = {}
for dense_units in (128, 1024):
    modl_func = tf_tutorial_keras_func_model(dense_size=dense_units)
    print("Training Keras Functional Model for dense layer of size:", dense_units)
    hist_func[dense_units] = modl_func.fit(x_train, y_train, epochs=16, batch_size=32, validation_split=0.5)
print_hist(hist_seq, title="tf-mnist Sequential Model & dense layer size")
print_hist(hist_func, title="tf-mnist Functional Model & dense layer size")