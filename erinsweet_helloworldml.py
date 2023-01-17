# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import random as rn



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



# plotly library

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import Pipeline

from scipy.stats import uniform



import itertools



import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



from keras.utils.np_utils import to_categorical

from keras.utils import np_utils



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

#from keras.layers import AvgPool2D, BatchNormalization, Reshape

from keras.optimizers import Adadelta, RMSprop, Adam

from keras.losses import categorical_crossentropy

from keras.wrappers.scikit_learn import KerasClassifier



import tensorflow as tf



import os

print(os.listdir("../input"))
img_rows, img_cols = 28, 28

np.random.seed(5)



def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_



def print_validation_report(y_true, y_pred):

    print("Classification Report")

    print(classification_report(y_true, y_pred))

    acc_sc = accuracy_score(y_true, y_pred)

    print("Accuracy : "+ str(acc_sc))

    

    return acc_sc



def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)

    #  square=True,

    plt.ylabel('true label')

    plt.xlabel('predicted label')

    

def plot_history_loss_and_acc(history_keras_nn):



    fig, axs = plt.subplots(1,2, figsize=(12,4))



    axs[0].plot(history_keras_nn.history['loss'])

    axs[0].plot(history_keras_nn.history['val_loss'])

    axs[0].set_title('model loss')

    axs[0].set_ylabel('loss')

    axs[0].set_xlabel('epoch')

    axs[0].legend(['train', 'validation'], loc='upper left')



    axs[1].plot(history_keras_nn.history['acc'])

    axs[1].plot(history_keras_nn.history['val_acc'])

    axs[1].set_title('model accuracy')

    axs[1].set_ylabel('accuracy')

    axs[1].set_xlabel('epoch')

    axs[1].legend(['train', 'validation'], loc='upper left')



    plt.show()
# read the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



y = train["label"]

X = train.drop(["label"],axis = 1)

X_test = test
# Normalization

X = X/255.0

X_test = X_test/255.0
# for best performance, especially of the NN classfiers,

# set mode = "commit"

mode = "edit"

mode = "commit"

#



if mode == "edit" :

    nr_samples = 1200



if mode == "commit" :    

    nr_samples = 30000



y_train=y[:nr_samples]

X_train=X[:nr_samples]

start_ix_val = nr_samples 

end_ix_val = nr_samples + int(nr_samples/3)

y_val=y[start_ix_val:end_ix_val]

X_val=X[start_ix_val:end_ix_val]



print("nr_samples train data:", nr_samples)

print("start_ix_val:", start_ix_val)

print("end_ix_val:", end_ix_val)
#  print first five samples

fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(10,6))

axs = axs.flatten()

for i in range(0,5):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[i].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[i].set_title(y[i])

plt.tight_layout()    

fig, ax = plt.subplots(figsize=(8,5))

g = sns.countplot(y)
import matplotlib.pyplot as plt

%matplotlib inline



li_idxs = []

for i in range(10):

    for nr in range(10):

        ix = y[y==nr].index[i]

        li_idxs.append(ix) 

        

fig, axs = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10,12))

axs = axs.flatten()

for n, i in enumerate(li_idxs):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[n].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[n].set_title(y[i])

plt.tight_layout() 
# NN Classifiers with Keras

y_train = to_categorical(y_train, 10)

y_val_10 = to_categorical(y_val, 10)
# Fully-Connected Neural Networks

def dense_model_0():

    model = Sequential()

    model.add(Dense(10, input_dim=784, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



model_dense_0 = dense_model_0()

model_dense_0.summary()
batchsize = int(nr_samples/15) 

model_dense_0.fit(X_train, y_train, epochs=50, batch_size=batchsize)

pred_val_dense0 = model_dense_0.predict_classes(X_val)

acc_fc0 = print_validation_report(y_val, pred_val_dense0)
acc_fc0 = print_validation_report(y_val, pred_val_dense0)
plot_confusion_matrix(y_val, pred_val_dense0)
# 1 hidden layer

def dense_model_1():

    model = Sequential()

    model.add(Dense(100, input_dim=784, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model_dense_1 = dense_model_1()

model_dense_1.summary()

history_dense_1 = model_dense_1.fit(X_train, y_train, validation_data=(X_val,y_val_10), epochs=50, batch_size=batchsize)

plot_history_loss_and_acc(history_dense_1)
pred_val_dense1 = model_dense_1.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_dense1)

print(classification_report(y_val, pred_val_dense1))

acc_fc1 = accuracy_score(y_val, pred_val_dense1)

print(acc_fc1)
# 2 hidden layers

def dense_model_2():

    model = Sequential()

    model.add(Dense(100, input_dim=784, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# 3 hidden layers

def dense_model_3():

    

    model = Sequential()  

    model.add(Dense(100, activation='relu', input_dim=784))

    model.add(Dense(200, activation='relu')) 

    model.add(Dense(100, activation='relu')) 

    model.add(Dense(10, activation='softmax'))

         

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    #model.compile(optimizer=RMSprop(lr=0.001),

    #         loss='categorical_crossentropy',

    #         metrics=['accuracy'])

    

    return model

# Convolutional Neural Networks, CNN



X_train.shape
X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_val = X_val.values.reshape(X_val.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)
batchsize = 128

epochs = 12

activation = 'relu'

adadelta = Adadelta()

loss = categorical_crossentropy



def cnn_model_1(activation):

    

    model = Sequential()

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=input_shape)) 

    

    model.add(Conv2D(64, (3, 3), activation=activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Flatten())



    model.add(Dense(128, activation=activation))

    model.add(Dropout(0.5))



    model.add(Dense(10, activation='softmax'))



    model.compile(loss=loss, optimizer=adadelta, metrics=['accuracy'])



    return model
model_cnn_1 = cnn_model_1(activation)

model_cnn_1.summary()
#model_cnn_1.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, verbose=1)

history_cnn_1 = model_cnn_1.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                   epochs=epochs, batch_size=batchsize, verbose=1)
plot_history_loss_and_acc(history_cnn_1)
pred_val_cnn1 = model_cnn_1.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_cnn1)

print(classification_report(y_val, pred_val_cnn1))

acc_cnn1 = accuracy_score(y_val, pred_val_cnn1)

print(acc_cnn1)
# CNN model 2



batch_size=90

epochs=30

def cnn_model_2(optimizer,loss):



    model = Sequential()



    model.add(Conv2D(32, (3, 3), padding = 'Same', activation="relu", input_shape=input_shape ))

    model.add(MaxPooling2D(pool_size = (2, 2)))



    model.add(Conv2D(32, (3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size = (2, 2)))



    model.add(Flatten())



    model.add(Dense(256, activation=activation))

    model.add(Dense(10, activation='softmax'))



    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy']) 



    return model
# predictions 

sample_submission = pd.read_csv('../input/sample_submission.csv')

if mode == "edit" :

    X = X[:nr_samples//2]

    y = y[:nr_samples//2]

    X_test = X_test[:nr_samples//2]

    sample_submission = sample_submission[:nr_samples//2]



# reshape for CNN

X = X.values.reshape(X.shape[0], img_rows, img_cols, 1)

X_test = X_test.values.reshape(X_test.shape[0], img_rows, img_cols, 1)

y = to_categorical(y, 10)



batchsize = 128

epochs = 12

model_cnn_1 = cnn_model_1('relu')

model_cnn_1.fit(X, y, epochs=epochs, batch_size=batchsize, verbose=0)



pred_test_cnn_1 = model_cnn_1.predict(X_test)

pred_test_cnn_1 = np.argmax(pred_test_cnn_1,axis=1)

result_cnn_1 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_cnn_1})

result_cnn_1.to_csv("subm_cnn_1.csv",index=False)
