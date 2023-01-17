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

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow
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

NO_EPOCHS_1 = 5

NO_EPOCHS_2 = 10

NO_EPOCHS_3 = 50

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
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE)
print("Train set rows: {}".format(train_df.shape[0]))

print("Test  set rows: {}".format(test_df.shape[0]))

print("Val   set rows: {}".format(val_df.shape[0]))
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("value", "value (train data)", train_df, size=3)
plot_count("value", "value (validation data)", val_df, size=3)
plot_count("value", "value (test data)", test_df, size=3)
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data_df["code"].values)

train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_df["code"].values)

print("Train set rows: {}".format(train_df.shape[0]))

print("Test  set rows: {}".format(test_df.shape[0]))

print("Val   set rows: {}".format(val_df.shape[0]))
plot_count("value", "value (train data)", train_df, size=3)
plot_count("value", "value (validation data)", val_df, size=3)
plot_count("value", "value (test data)", test_df, size=3)
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
model1=Sequential()

model1.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), activation='relu', padding='same'))

model1.add(MaxPool2D(MAX_POOL_DIM))

model1.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

model1.add(Flatten())

model1.add(Dense(y_train.columns.size, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
train_model1 = model1.fit(X_train, y_train,

                  batch_size=BATCH_SIZE,

                  epochs=NO_EPOCHS_1,

                  verbose=1,

                  validation_data=(X_val, y_val))
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



plot_accuracy_and_loss(train_model1)
score = model1.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def test_accuracy_report(model):

    predicted = model.predict(X_test)

    test_predicted = np.argmax(predicted, axis=1)

    test_truth = np.argmax(y_test.values, axis=1)

    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 

    test_res = model.evaluate(X_test, y_test.values, verbose=0)

    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
test_accuracy_report(model1)
model2=Sequential()

model2.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))

model2.add(MaxPool2D(MAX_POOL_DIM))

# Add dropouts to the model

model2.add(Dropout(0.4))

model2.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

# Add dropouts to the model

model2.add(Dropout(0.4))

model2.add(Flatten())

model2.add(Dense(y_train.columns.size, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()
train_model2  = model2.fit(X_train, y_train,

                  batch_size=BATCH_SIZE,

                  epochs=NO_EPOCHS_2,

                  verbose=1,

                  validation_data=(X_val, y_val))
plot_accuracy_and_loss(train_model2)
test_accuracy_report(model2)
annealer3 = LearningRateScheduler(lambda x: 1e-3 * 0.995 ** (x+NO_EPOCHS_3))

earlystopper3 = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

checkpointer3 = ModelCheckpoint('best_model_3.h5',

                                monitor='val_acc',

                                verbose=VERBOSE,

                                save_best_only=True,

                                save_weights_only=True)
model3=Sequential()

model3.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))

model3.add(MaxPool2D(MAX_POOL_DIM))

# Add dropouts to the model

model3.add(Dropout(0.4))

model3.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))

# Add dropouts to the model

model3.add(Dropout(0.4))

model3.add(Flatten())

model3.add(Dense(y_train.columns.size, activation='softmax'))

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.summary()
train_model3  = model3.fit(X_train, y_train,

                  batch_size=BATCH_SIZE,

                  epochs=NO_EPOCHS_3,

                  verbose=1,

                  validation_data=(X_val, y_val),

                  callbacks=[earlystopper3, checkpointer3, annealer3])
plot_accuracy_and_loss(train_model3)
test_accuracy_report(model3)