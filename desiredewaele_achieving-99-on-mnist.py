import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import optimizers

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
warnings.filterwarnings('ignore')
# CONSTANTS
SIZE = 28
LABELS = 10
CHANNELS = 1
def digits(data, labels=None, random=False, xy=None):
    if xy is None: x, y = 20, min(10, len(data) // 20 +1)
    else: x, y = xy
    fig, ax = plt.subplots(y, x, figsize = (x, y))
    if x==1: indeces = np.arange(y)
    elif y==1: indeces = np.arange(x)
    else: indeces = [(i,j) for i in np.arange(y) for j in np.arange(x)]
    for i, index in enumerate(indeces[:len(data)]):
        if random: i = np.random.randint(0, len(data))
        ax[index].matshow(data.reshape(-1, SIZE, SIZE)[i], cmap=cm.gray_r)
        if labels: ax[index].set_title("Label: {}".format(labels[i]))
    for index in indeces: ax[index].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
def plotter(history):
    at, av, lt, lv = zip(*history)
    fig = plt.figure(figsize=(15, 8)); ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)

    ax1.plot(np.arange(0, len(at), 1), at,".-", color='#2A6EA6', label="Training: {0:.2f}%".format(at[-1]))
    ax1.plot(np.arange(0, len(av), 1), av,".-", color='#FFA933', label="Validation: {0:.2f}%".format(av[-1]))
    ax1.grid(True); ax1.legend(loc="lower right"); ax1.set_title("Accuracy per epoch")

    ax2.plot(np.arange(0, len(lt), 1), lt,".-", color='#2A6EA6', label="Training: {0:.2f}".format(lt[-1]))
    ax2.plot(np.arange(0, len(lv), 1), lv,".-", color='#FFA933', label="Validation: {0:.2f}".format(lv[-1]))
    ax2.grid(True); ax2.legend(loc="upper right"); ax2.set_title("Cost per epoch")
    plt.show()
def plotCertainty(data, predictions, probabilities):
    fig1, ax1 = plt.subplots(1, 10, figsize = (20, 2))
    fig2, ax2 = plt.subplots(1, 10, figsize = (20, 2))
    for t in np.arange(10):
        ax1[t].matshow(data.reshape(-1, SIZE, SIZE)[t], cmap=cm.gray_r)
        ax1[t].get_yaxis().set_visible(False); ax1[t].get_xaxis().set_visible(False)
        ax2[t].set_title("Prediction: {}".format(predictions[t]))
        ax2[t].plot(probabilities[t], lw=2)
        ax2[t].set_title("Prediction: {}".format(predictions[t]))
        ax2[t].get_yaxis().set_visible(False); ax2[t].set_ylim([0, 1]); ax2[t].grid(True)
    plt.show()
data = pd.read_csv('/kaggle/input/train.csv')
test = pd.read_csv('/kaggle/input/test.csv')
labels = data.pop('label')
data.tail()
labels.head()
digits(data.values, random=True)
print('Training set', data.shape, labels.shape)
print('Testing set', test.shape)
data = data.values.reshape(-1, SIZE, SIZE, CHANNELS)
testX = test.values.reshape(-1, SIZE, SIZE, CHANNELS)
labels = pd.get_dummies(labels).values
print('Training set', data.shape, labels.shape)
print('Testing set', testX.shape)
from sklearn.model_selection import train_test_split
trainX, validX, trainY, validY = train_test_split(data, labels, train_size=32000, random_state=100)
print('Training set:', trainX.shape, trainY.shape)
print('Validation set:', validX.shape, validY.shape)
print('Testing set:', testX.shape)
trainAugmenter = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=(0.92, 1.08),
        horizontal_flip=False)
validAugmenter = ImageDataGenerator()
testAugmenter = ImageDataGenerator()
BATCHSIZE = 128
trainGenerator = trainAugmenter.flow(trainX, trainY, BATCHSIZE)
validGenerator = validAugmenter.flow(validX, validY, BATCHSIZE)
testGenerator = testAugmenter.flow(testX, None, BATCHSIZE)
for batchX, batchY in trainGenerator:
    digits(batchX)
    break
NUMFILTERS = 16
FILTER = 3
ACTIVATION = 'tanh'
DROPOUT = 0.5
model = Sequential([
    Conv2D(NUMFILTERS, (FILTER, FILTER), padding='same', activation=ACTIVATION, input_shape=(trainX.shape[1:])),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(NUMFILTERS*2, (FILTER, FILTER), padding='same', activation=ACTIVATION),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(NUMFILTERS*4, (FILTER, FILTER), padding='same', activation=ACTIVATION),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(NUMFILTERS*8, (FILTER, FILTER), padding='same', activation=ACTIVATION),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dropout(DROPOUT),
    Dense(trainY.shape[1]*6, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(trainY.shape[1]*2, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(trainY.shape[1], activation='softmax'),
])
model.compile(optimizer=optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
EPOCHS = 50
model.fit_generator(trainGenerator, epochs=EPOCHS, validation_data=validGenerator)
x = np.arange(EPOCHS)+1
history = model.history.history

data = [
    go.Scatter(x=x, y=history["acc"], name="Train Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["val_acc"], name="Valid Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["loss"], name="Train Loss", marker=dict(size=5)),
    go.Scatter(x=x, y=history["val_loss"], name="Valid Loss", marker=dict(size=5))
]
layout = go.Layout(title="Model Training Evolution", font=dict(family='Palatino'), 
                   xaxis=dict(title='Epoch', range=[0,EPOCHS]),
                   yaxis1=dict(title="Loss", domain=[0, 0.45]), 
                   yaxis2=dict(title="Accuracy", domain=[0.55, 1]))
py.iplot(go.Figure(data=data, layout=layout), show_link=False)
print(model.evaluate(trainX, trainY))
print(model.evaluate(validX, validY))
validProbs = model.predict(validX)
validPreds = validProbs.argmax(axis=1)
validTruth = validY.argmax(axis=1)
confusionmatrix = pd.crosstab(validPreds, validTruth, normalize=False)
trace = go.Heatmap(x=np.arange(10), y=np.arange(10), z=confusionmatrix.values,
                   colorscale = [[0,"#ffffff"], [0.1,"#000000"], [1,"#000000"]],
                   colorbar = dict(ticksuffix='', thickness=15, len=1))
layout = go.Layout(font=dict(family='Palatino'), height=500, width=600,
                   margin = go.layout.Margin(l=50, r=30, b=30, t=50), 
                   xaxis=dict(title="Truth", dtick=1, side='top'), 
                   yaxis=dict(title="Prediction", dtick=1, autorange='reversed'))
py.iplot(go.Figure(data=[trace], layout=layout), show_link=False)
digits(validX[(validTruth==4)&(validPreds==9)])
digits(validX[(validTruth==9)&(validPreds==8)])
order = (-validProbs.max(axis=1)).argsort()
plotCertainty(validX[order[-10:]], validPreds[order[-10:]], validProbs[order[-10:]])
plotCertainty(validX[order[:10]], validPreds[order[:10]], validProbs[order[:10]])
probabilities = model.predict(testX)
predictions = probabilities.argmax(axis=1)
submission = pd.DataFrame({"ImageId":np.arange(1, len(predictions)+1), "Label":predictions})
submission.to_csv("submissionMnist.csv", index=False)
submission.head()