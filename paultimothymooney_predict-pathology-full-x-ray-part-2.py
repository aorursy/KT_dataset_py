import pandas as pd

import numpy as np

import os

import itertools

from glob import glob

import random

import matplotlib.pylab as plt

import cv2

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

import keras

from keras.models import Sequential, model_from_json

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

from keras import callbacks

from keras.utils.vis_utils import plot_model

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.utils.io_utils import HDF5Matrix

from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from sklearn.metrics import confusion_matrix

from sklearn.utils import class_weight

from keras.applications.mobilenet import MobileNet

from keras.callbacks import Callback,ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import h5py

%matplotlib inline
disease_vec_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',

 'Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

dict_characters = {0:'Atelectasis',1:'Cardiomegaly',2:'Consolidation',3:'Edema',4:'Effusion',5:'Emphysema',6:'Fibrosis',

 7:'Hernia',8:'Infiltration',9:'Mass',10:'Nodule',11:'Pleural_Thickening',12:'Pneumonia',13:'Pneumothorax'}

h5_path = '../input/create-a-mini-xray-dataset-equalized/chest_xray.h5'

disease_vec_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',

 'Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

disease_vec = []

with h5py.File(h5_path, 'r') as h5_data:

    all_fields = list(h5_data.keys())

    for c_key in all_fields:

        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)

    for c_key in disease_vec_labels:

        disease_vec += [h5_data[c_key][:]]

disease_vec = np.stack(disease_vec,1)

print('Disease Vec:', disease_vec.shape)
img_ds = HDF5Matrix(h5_path, 'images')

split_idx = img_ds.shape[0]//2

train_ds = HDF5Matrix(h5_path, 'images', end = split_idx)

test_ds = HDF5Matrix(h5_path, 'images', start = split_idx)

train_dvec = disease_vec[0:split_idx]

test_dvec = disease_vec[split_idx:]

print('Train Shape', train_ds.shape, 'test shape', test_ds.shape)



X = img_ds

y = disease_vec

X = np.asarray(X)

y = np.asarray(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# Reduce Sample Size for DeBugging

X_train = X_train[0:500000] 

Y_train = Y_train[0:500000]

X_test = X_test[0:200000] 

Y_test = Y_test[0:200000]



print("X_train",X_train.shape)

print("Y_train",Y_train.shape)

print("X_test",X_test.shape)

print("Y_test",Y_test.shape)
class MetricsCheckpoint(Callback):

    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):

        super(MetricsCheckpoint, self).__init__()

        self.savepath = savepath

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        np.save(self.savepath, self.history)



def plotKerasLearningCurve():

    plt.figure(figsize=(10,5))

    metrics = np.load('logs.npy')[()]

    filt = ['acc'] # try to add 'loss' to see the loss learning curve

    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):

        l = np.array(metrics[k])

        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')

        x = np.argmin(l) if 'loss' in k else np.argmax(l)

        y = l[x]

        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')

        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   

    plt.legend(loc=4)

    plt.axis([0, None, None, None]);

    plt.grid()

    plt.xlabel('Number of epochs')



def plot_learning_curve(history):

    plt.figure(figsize=(8,8))

    plt.subplot(1,2,1)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./accuracy_curve.png')

    #plt.clf()

    # summarize history for loss

    plt.subplot(1,2,2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./loss_curve.png')



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (5,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

def runCNNconfusion(a,b,c,d):

    # Set the CNN model 

    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    batch_size = 128

    num_classes = 14

    epochs = 4 

        # input image dimensions

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 1)

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = input_shape))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Flatten())

    #model.add(Dense(1024, activation = "relu"))

    #model.add(Dropout(0.5))

    model.add(Dense(512, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation = "softmax"))

    # Define the optimizer

    optimizer = RMSprop(lr=0.001, decay=1e-6)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

#         horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    datagen.fit(a)

    history = model.fit_generator(datagen.flow(a,b, batch_size=32),

                        steps_per_epoch=len(a) / 32, epochs=epochs,  validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])

    # , class_weight = class_weight

    score = model.evaluate(c,d, verbose=0) 

    print('\nKeras CNN - accuracy:', score[1],'\n')

    plot_learning_curve(history)

    plt.show()

    plotKerasLearningCurve()

    plt.show()

    Y_pred = model.predict(c)

    #print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(Y_pred, axis=1), target_names=list(dict_characters.values())), sep='')    

    Y_pred_classes = np.argmax(Y_pred,axis = 1) 

    Y_true = np.argmax(d,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 

    plt.show()

runCNNconfusion(X_train, Y_train, X_test, Y_test)
def MOBILENET(a,b,c,d):

    raw_model = MobileNet(input_shape=(None, None, 1), include_top = False, weights = None)

    full_model = Sequential()

    full_model.add(AveragePooling2D((2,2), input_shape = img_ds.shape[1:]))

    full_model.add(BatchNormalization())

    full_model.add(raw_model)

    full_model.add(Flatten())

    full_model.add(Dropout(0.5))

    full_model.add(Dense(64))

    full_model.add(Dense(disease_vec.shape[1], activation = 'sigmoid'))

    full_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    full_model.summary()

    file_path="weights.best.hdf5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=3)

    callbacks_list = [checkpoint, early, MetricsCheckpoint('logs')] #early

    history = full_model.fit(X_train, Y_train, 

                   validation_data = (X_test, Y_test),

                   epochs=10, 

                   verbose = True,

                  shuffle = 'batch',

                  callbacks = callbacks_list)

    plot_learning_curve(history)

    plt.show()

    plotKerasLearningCurve()

    plt.show()

    model = full_model

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('\nMobileNet - accuracy:', score[1],'\n')

    y_pred = model.predict(X_test)

    map_characters = dict_characters

    Y_pred_classes = np.argmax(y_pred,axis=1) 

    Y_true = np.argmax(Y_test,axis=1)

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 

    plt.show()

MOBILENET(X_train, Y_train, X_test, Y_test)