# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

import seaborn as sns
from matplotlib import pyplot as plt
# numpy random number geneartor seed
# for reproducibility
np.random.seed(123)

# set plot rc parameters
# jtplot.style(grid=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#232323'
#plt.rcParams['axes.edgecolor'] = '#FFFFFF'
plt.rcParams['figure.figsize'] = 10, 7
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.framealpha'] = 0.2
plt.rcParams['text.color'] = '#666666'
plt.rcParams['axes.labelcolor'] = '#666666'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['ytick.labelsize'] = 14

# plt.rcParams['font.size'] = 16

sns.color_palette('dark')
%matplotlib inline
dftrain = pd.read_csv('../input/digit-recognizer/train.csv')
dftest = pd.read_csv('../input/digit-recognizer/test.csv')
dftrain.head()
dftrain.shape, dftest.shape
# plot histogram of digits
fig = plt.figure(figsize=[10,7])
sns.countplot(dftrain['label'], color=sns.xkcd_rgb['greenish cyan'])
plt.title('Histogram of digits')
plt.show()
# generate random list of indices of data point
list_idx = [random.randint(0,1000) for a in range(10)]
# plot randomly chosen data points
fig, axs = plt.subplots(5,2, figsize=[25,25])
axs = axs.flatten()
for i in range(0,10,2):
    # get x and y values from image
    y = dftrain.iloc[list_idx[i]]['label']
    x = dftrain.iloc[list_idx[i]][1:].values
    # plot flattened image (1D)
    axs[i].plot(x)
    axs[i].set_xlabel('#pixel')
    axs[i].set_ylabel('intensity')
    # axs[i].set_title(str(y))
    # plot 2D image
    axs[i+1].imshow(x.reshape(28,28))
    axs[i+1].set_title('image of number' + str(y))
    
plt.show()
# split data into train and test set
Xtrain, Xcv, Ytrain, Ycv = train_test_split(dftrain.values[:,1:], dftrain.values[:,0], random_state=16, test_size=0.2)
# one hot encode label matrix
def ohe_y(y):
    return to_categorical(y, 10)

# pre-process image pixel
def pre_process_x(x):
    # reshape array as 28X28 matrix
    out_x = x.reshape(-1, 28, 28, 1)
    # normalize 
    out_x = out_x.astype('float32')
    out_x /= 255
    return out_x

Xtrain = pre_process_x(Xtrain)
Xcv = pre_process_x(Xcv)
Xtest = pre_process_x(dftest.values)
Ytrain= ohe_y(Ytrain)
Ycv = ohe_y(Ycv)
# print meta data
print('Xtrain shape\t\t', Xtrain.shape, '\n',
     'Xcv shape\t\t', Xcv.shape, '\n',
     'Xtest shape\t\t', Xtest.shape, '\n',
     'Ytrain shape\t\t', Ytrain.shape, '\n',
     'Ycv shape\t\t', Ycv.shape, '\n')
# initiate sequential model
model = Sequential()
# add convolutional layer
# 16 sliding windows each of 3X3 size
# default step is 1X1
model.add(Conv2D(filters = 16,
                 kernel_size = (3, 3),
                 activation='relu',
                 input_shape = (28, 28,1)))
# let's print shape of output of first layer
print('shape of output of first convolution layer\t\t', model.output_shape, '\n\n')
# add batch normalization to normalize output of the layer
model.add(BatchNormalization())
# add another convolutional layer
model.add(Conv2D(filters = 16,
                 kernel_size = (3, 3),
                 activation='relu'))
# let's print shape of output of second layer
print('shape of output of secon convolution layer\t\t', model.output_shape, '\n\n')
# batchnormalize
model.add(BatchNormalization())
# add maxpooling layer
# this layer picks max value for every 2X2 window
model.add(MaxPool2D(pool_size=(2,2)))
# let's print shape of output of maxpool layer
print('shape of output of first maxpool layer\t\t', model.output_shape, '\n\n')
# add dropout layer
# it acts as a regularizer
model.add(Dropout(0.25))
# repeat above sequence once more
model.add(Conv2D(filters = 32,
                 kernel_size = (3, 3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32,
                 kernel_size = (3, 3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# now we'll add two dense layer
# just like layer of ANN
print('shape of output before flattening data\t\t', model.output_shape, '\n\n')
model.add(Flatten())
print('shape of output after flattening data\t\t', model.output_shape, '\n\n')
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
# let's print shape of output of dense layers
print('shape of output of dense layers\t\t', model.output_shape, '\n\n')
model.add(Dropout(0.5))
# finally add a softmax layer which will predict probability of each class
model.add(Dense(10, activation='softmax'))
# print model summary
model.summary()

# compile model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
# train model
model_out = model.fit(Xtrain,
                      Ytrain,
                     batch_size=128,
                     epochs=40,
                     validation_data=(Xcv[:500], Ycv[:500]))
# evaluate model performance
loss, acc = model.evaluate(Xcv, Ycv, verbose=0)
# Print loss and accuracy of model
loss, acc
# plot model accuracy and error with respect to each epoch
fig, axs = plt.subplots(1,2, figsize=[15,7])
axs.flatten()
# plot accuracy
axs[0].plot(model_out.history['accuracy'], sns.xkcd_rgb['greenish cyan'], label='training')
axs[0].plot(model_out.history['val_accuracy'], sns.xkcd_rgb['red pink'], label='cross validation')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(loc='best')
# plot loss
axs[1].plot(model_out.history['loss'], sns.xkcd_rgb['greenish cyan'], label='training')
axs[1].plot(model_out.history['val_loss'], sns.xkcd_rgb['red pink'], label='cross validation')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(loc='best')
plt.show()
# plot confusion matrix
def plot_confusion_matrix(model, Xtrain, Xtest, Ytrain, Ytest):
    
    # get predicted values
    Ytrain_pred = model.predict(Xtrain)
    Ytest_pred = model.predict(Xtest)
    
    # flatten ypredicted and ytrue
    Ytrain = np.argmax(Ytrain, axis=1)
    Ytest = np.argmax(Ytest, axis=1)
    Ytrain_pred = np.argmax(Ytrain_pred, axis=1)
    Ytest_pred = np.argmax(Ytest_pred, axis=1)

    # confusion matrix
    fig, axs = plt.subplots(1,2,
                            figsize=[15,8])
    axs = axs.flatten()
    
    axs[0].set_title('Training data')
    # axs[0].set_xlabel('Predicted label')
    # axs[0].set_ylabel('True label')
    axs[1].set_title('Test data')
    # axs[1].set_xlabel('Predicted label')
    # axs[1].set_ylabel('True label')
    
    fig.text(0.3, 0.1, 'Predicted label', ha='center', fontsize=14)
    fig.text(0.72, 0.1, 'Predicted label', ha='center', fontsize=14)
    fig.text(0.08, 0.5, 'True label', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.5, 'True label', va='center', rotation='vertical', fontsize=14)
    
    sns.heatmap(confusion_matrix(Ytrain,Ytrain_pred),
                    annot=True,
                    xticklabels=list(range(10)),
                    yticklabels=list(range(10)),
                    fmt="d",
                    cmap='coolwarm',
                    square=True,
                    cbar=False,
                    ax=axs[0])
    
    sns.heatmap(confusion_matrix(Ytest,Ytest_pred),
                    annot=True,
                    xticklabels=list(range(10)),
                    yticklabels=list(range(10)),
                    fmt="d",
                    cmap='coolwarm',
                    square=True,
                    cbar=False,
                    ax=axs[1])
    plt.show()
    
    return

plot_confusion_matrix(model,Xtrain,Xcv,Ytrain,Ycv)
# get prediction of test data
Ytest = model.predict(Xtest)
Ytest = np.argmax(Ytest,axis=1)
# get id column
ids = list(range(1,dftest.shape[0]+1))
# put that in pandas data frame
submission = pd.DataFrame({'ImageId':ids, 'Label':Ytest})
submission.head()
# save submission file
submission.to_csv('submission.csv')
