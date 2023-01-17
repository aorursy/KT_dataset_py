import os
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
from keras.layers import Convolution2D, Activation, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
def tf_tutorial_model(dropout=0.4, dense_size=1024, lr=0.0005, act='relu'):
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(32, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(64, (5,5), padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    model.add(Flatten())
    # first dense layer:
    if dense_size > 0:
        model.add(Dense(dense_size))
        model.add(Activation(act))
        if dropout > 0:
            model.add(Dropout(dropout))
    # final (dense) layer:
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def lenet_5_model(lr=0.1, act='sigmoid'):  # see Fig 2 in LeNet-5/Lecun.
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(6, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(16, (5,5), padding='valid', data_format='channels_first'))
    model.add(Activation(act))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    model.add(Flatten())
    # full connection i
    model.add(Dense(120))
    model.add(Activation(act))
    # full connection ii
    model.add(Dense(84))
    model.add(Activation(act))
    # final (dense) layer:
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# little plotter helper to see results. various dense layer data is plotted simultaneously against the epochs.
def plotxy(hist, title="tf-mnist & LeNet-5", ytitle="accuracy", xtitle="epoch", legend=()):
    fig = plt.figure(figsize=(15, 10)) #change dpi to get bigger resolution..
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
        legend.append(str(d))  # legends e.g. d-128, d-256 denote the dense layer size.
    return((plot_data, legend))
        
def list_val_acc_fm_history(hist, column='Dense layer'):
    fmt_str='{0:11} :   {1:6}'
    print(fmt_str.format(column, id))
    for d in sorted(hist.keys()):
        print(fmt_str.format(str(d), round(hist[d].history[id][-1], 3)))
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
x_train = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(train_data['label'])
print("Original Data-set, Variance:", round(np.var(x_train), 3), "  Mean:", round(np.mean(x_train), 3),   "  Min:", round(np.min(x_train), 3), "  Max:", round(np.max(x_train), 3))
n_train = x_train/ np.sqrt(np.var(x_train))
n_train = n_train - np.mean(n_train)
print("Rescaled Data-set, Variance:", round(np.var(n_train), 3), "  Mean:", round(np.mean(n_train), 3),   "  Min:", round(np.min(n_train), 3), "  Max:", round(np.max(n_train), 3))
hist = {}
for dense_units in (128, 1024): 
    modld = tf_tutorial_model(dense_size=dense_units, lr=0.1)
    print("Training for dense layer of size:", dense_units)
    hist["TF-" + str(dense_units)+"-norm"] = modld.fit(n_train, y_train, epochs=64, batch_size=32, validation_split=0.5)
modl5 = lenet_5_model(lr=1.0) # see Fig 2 in LeNet-5/Lecun.
print("Training LeNet-5 model")
hist["LeNet-5-norm"] = modl5.fit(n_train, y_train, epochs=64, batch_size=32, validation_split=0.5)   # normalize 0--255 to -0.1--1.175
id = 'val_acc'
(pl_data,legnd) = get_data_n_legend(hist, id)
plotxy(pl_data, title="tf-mnist (d-1024 & d-128) and lenet-5 (d-120) validation-accuracy comparison:", xtitle='epoch', legend=legnd, ytitle=id)
list_val_acc_fm_history(hist)