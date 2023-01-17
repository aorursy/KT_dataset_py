import pandas as pd

import numpy as np

import sys

import os

import random

from pathlib import Path

import imageio

import skimage

import skimage.io

import skimage.transform

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import scipy

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical

import tensorflow_addons as tfa

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
IMAGE_PATH = '..//input//chinese-mnist//data//data//'

IMAGE_WIDTH = 64

IMAGE_HEIGHT = 64

IMAGE_CHANNELS = 1

RANDOM_STATE = 42

TEST_SIZE = 0.2

VAL_SIZE = 0.2

CONV_2D_DIM_1 = 16

CONV_2D_DIM_2 = 16

CONV_2D_DIM_3 = 32

CONV_2D_DIM_4 = 64

MAX_POOL_DIM = 2

KERNEL_SIZE = 3

BATCH_SIZE = 32

NO_EPOCHS = 50

DROPOUT_RATIO = 0.5

PATIENCE = 5

VERBOSE = 1
os.listdir("..//input//chinese-mnist")
data_df=pd.read_csv('..//input//chinese-mnist//chinese_mnist.csv')
data_df.shape
data_df.sample(100).head()
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data(data_df)
image_files = list(os.listdir(IMAGE_PATH))

print("Number of image files: {}".format(len(image_files)))
def create_file_name(x):

    

    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"

    return file_name
data_df["file"] = data_df.apply(create_file_name, axis=1)
file_names = list(data_df['file'])

print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
def read_image_sizes(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    return list(image.shape)
m = np.stack(data_df['file'].apply(read_image_sizes))

df = pd.DataFrame(m,columns=['w','h'])

data_df = pd.concat([data_df,df],axis=1, sort=False)
data_df.head()
print(f"Number of suites: {data_df.suite_id.nunique()}")

print(f"Samples: {data_df.sample_id.unique()}")
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data_df["code"].values)
train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_df["code"].values)
print("Train set rows: {}".format(train_df.shape[0]))

print("Test  set rows: {}".format(test_df.shape[0]))

print("Val   set rows: {}".format(val_df.shape[0]))
def read_image(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 1), mode='reflect')

    return image[:,:,:]
def categories_encoder(dataset, var='character'):

    X = np.stack(dataset['file'].apply(read_image))

    y = pd.get_dummies(dataset[var], drop_first=False)

    return X, y
X_train, y_train = categories_encoder(train_df)

X_val, y_val = categories_encoder(val_df)

X_test, y_test = categories_encoder(test_df)
model=Sequential()

model.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))

model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model.add(MaxPool2D(MAX_POOL_DIM))

model.add(Dropout(DROPOUT_RATIO))

model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model.add(Dropout(DROPOUT_RATIO))

model.add(Flatten())

model.add(Dense(y_train.columns.size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))

earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

checkpointer = ModelCheckpoint('best_model.h5',

                                monitor='val_accuracy',

                                verbose=VERBOSE,

                                save_best_only=True,

                                save_weights_only=True)
train_model  = model.fit(X_train, y_train,

                  batch_size=BATCH_SIZE,

                  epochs=NO_EPOCHS,

                  verbose=1,

                  validation_data=(X_val, y_val),

                  callbacks=[earlystopper, checkpointer, annealer])
def create_trace(x,y,ylabel,color):

        trace = go.Scatter(

            x = x,y = y,

            name=ylabel,

            marker=dict(color=color),

            mode = "markers+lines",

            text=x

        )

        return trace

    

def plot_accuracy_and_loss(train_model):

    hist = train_model.history

    acc = hist['accuracy']

    val_acc = hist['val_accuracy']

    loss = hist['loss']

    val_loss = hist['val_loss']

    epochs = list(range(1,len(acc)+1))

    #define the traces

    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")

    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")

    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")

    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")

    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',

                                                             'Training and validation loss'))

    #add traces to the figure

    fig.append_trace(trace_ta,1,1)

    fig.append_trace(trace_va,1,1)

    fig.append_trace(trace_tl,1,2)

    fig.append_trace(trace_vl,1,2)

    #set the layout for the figure

    fig['layout']['xaxis'].update(title = 'Epoch')

    fig['layout']['xaxis2'].update(title = 'Epoch')

    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])

    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])

    #plot

    iplot(fig, filename='accuracy-loss')



plot_accuracy_and_loss(train_model)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def test_accuracy_report(model):

    predicted = model.predict(X_test)

    test_predicted = np.argmax(predicted, axis=1)

    test_truth = np.argmax(y_test.values, axis=1)

    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 

    test_res = model.evaluate(X_test, y_test.values, verbose=0)

    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
test_accuracy_report(model)
model_optimal = model

model_optimal.load_weights('best_model.h5')

score = model_optimal.evaluate(X_test, y_test, verbose=0)

print(f'Best validation loss: {score[0]}, accuracy: {score[1]}')



test_accuracy_report(model_optimal)