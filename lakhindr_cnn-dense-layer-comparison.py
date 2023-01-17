import os
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
def tf_tutorial_model(dropout=0.4, dense_size=1024):
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(32, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(64, (5,5), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # first dense layer:
    model.add(Flatten())
    model.add(Dense(dense_size))
    model.add(Activation('relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    # final (dense) layer:
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# little plotter helper to see results. various dense layer data is plotted simultaneously against the epochs.
def plotxy(hist, title="tf-mnist", ytitle="val_acc", xtitle="epoch", legend=()):
    fig = plt.figure(figsize=(10,6)) #change dpi to get bigger resolution..
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
        # if d < 32: continue
        plot_data.append(hist[d].history[id])
        legend.append('d-' + str(d))  # legends e.g. d-128, d-256 denote the dense layer size.
    return((plot_data, legend))
        
# make a graph of the history gathered..
def plot_hist_graph(hist, title="tf-mnist dense layers", id = "val_acc"):
    (pl_data, legnd) = get_data_n_legend(hist, id)
    plotxy(pl_data, title, xtitle='epoch', legend=legnd, ytitle=id)
def show_result(hist, id='val_acc', prefix='d-'):
    fmt_str='{0:6} :       {1:7}        {2:8}'
    print(fmt_str.format('size', id + ' %', 'Training acc %'))
    for d in sorted(hist.keys()):
        print(fmt_str.format(prefix + str(d), round(hist[d].history[id][-1] * 100, 1), round(hist[d].history['acc'][-1] *100, 1)))
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
x_train = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(train_data['label'])
x_train = x_train/255.0

hist = {}
split=0.75  # use 75% to train, 25% to validate.
tlenx = int(x_train.shape[0] * split)  #training data length
vdata= (x_train[tlenx:], y_train[tlenx:])
trn_x, trn_y = (x_train[0:tlenx], y_train[0:tlenx])
for dense_units in (128, 1024):
    modl = tf_tutorial_model(dense_size=dense_units)
    print("Training for dense layer of size:", dense_units)
    datagenT = ImageDataGenerator(featurewise_center=False, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, data_format='channels_first')
    dfr = datagenT.flow(trn_x, trn_y, batch_size=32)
    hist[dense_units] = modl.fit_generator(dfr, epochs=32, validation_data=vdata, verbose=1)
plot_hist_graph(hist)
show_result(hist)