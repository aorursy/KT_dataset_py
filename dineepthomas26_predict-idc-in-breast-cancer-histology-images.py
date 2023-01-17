import pandas as pd
import numpy as np
import argparse
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import cv2
from scipy.misc import imresize, imread
import sklearn
from sklearn.utils import class_weight
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, roc_auc_score,auc,roc_curve
import keras
from keras import layers
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json,load_model,Model
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import (Dense, Dropout,concatenate, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D,Input,GlobalAveragePooling2D,
GlobalMaxPooling2D,ZeroPadding2D,AveragePooling2D,Reshape,Convolution2D)
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from time import time

class_names = ['IDC_0', 'IDC_1']
roc_=[]

def main():
#     path = parsed_args.loc_data
    Tx_train, Ty_train = proc_images("train")
    print("out of proc images")
    df = pd.DataFrame()
    df["images"] = Tx_train
    df["labels"] = Ty_train
    X2 = df["images"]
    Y2 = df["labels"]
    X2 = np.array(X2)
    summary(X2, Y2) # description for training dataset
    X = np.array(Tx_train)
    X = X / 255.0
    print("splitting start")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Ty_train, test_size=0.2)
    Y_trainHot = to_categorical(Y_train, num_classes=2)
    Y_testHot = to_categorical(Y_test, num_classes=2)
    class_weigh = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    #
    # Vx_test, Vy_test = proc_images("test") # dataset for testing
    # df2 = pd.DataFrame()
    # df2["images"] = Vx_test
    # df2["labels"] = Vy_test
    # X22 = df["images"]
    # Y22 = df["labels"]
    # X22 = np.array(X22)
    # summary(X22, Y22) # description for test dataset
    # VX_test = np.array(Vx_test)
    # VX_test = VX_test / 255.0
    # # model calling starts now
    if(True):
        roc_=[]
        print("vgg16 executing " )
        vgg16_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh)

    if(True):
        roc_=[]
        print("vgg19 executing " )
        vgg19_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh)

    if(True):
        roc_=[]
        print("parallel executing " )
#         parallel_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh)

    if(True):
        roc_=[]
        print("non parallel executing "  )
        nonParallel_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh)

def proc_images(cat):
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    lowerIndex=0
    if(cat=='test'):
        imagePatches = glob( '../input/IDC_regular_ps50_idx5/1025*/**/*.png', recursive=True)
        print("test ", len(imagePatches))
    else:
        imagePatches = glob( '../input/IDC_regular_ps50_idx5/*/**/*.png', recursive=True)
        print("train imagepatches over", len(imagePatches))
    class0 = '*class0.png'
    class1 = '*class1.png'
    classZero = fnmatch.filter(imagePatches, class0)
    classOne = fnmatch.filter(imagePatches, class1)
    upperIndex = len(imagePatches)
    print("entering loop")
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x, y

def summary(a, b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) Images  : {}'.format(np.sum(b == 0)))
    print('Number of IDC(+) Images  : {}'.format(np.sum(b == 1)))
    print('Percentage of positive images  : {:.2f}%'.format(100 * np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))

class MetricsCheckpoint(Callback):

    def __init__(self, savepath, training_data, validation_data):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self,logs={}):
        return

    def on_train_end(self,logs={}):
        return

    def on_epoch_begin(self,epoch,logs={}):
        return

    # roc calculation starts for each epoch
    def on_epoch_end(self,epoch,logs={}):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)
        y_predict = self.model.predict(self.x_val)
        roc_valid = roc_auc_score(self.y_val, y_predict)
        # print("roc_valid ",roc_valid)
        roc_.append(roc_valid)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def plot_confusion_matrix (cm,classes,normalize,title,category):
    cmap=plt.cm.Blues
    plt.figure(figsize=(5, 5))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=90)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(category+'_'+'_'+title+'_BN_model_CM.pdf', bbox_inches='tight')

def plot_final(images,labels_true,class_names,model_out,form):
    # randomly plotting 9 images from test set
    fig =plt.figure()
    i=j=0
    print(len(images))
    for data,num in zip(images[:6],labels_true[:6]):
        img_num = num
        img_data = data

        y = fig.add_subplot(5,5,i*1+2)

        orig = img_data
        data = img_data.reshape(50,50,3)

        if model_out[i] == 1:
            str_label= "+"
        else:
            str_label= "-"
        if img_num == 1:
            str_label_orig= "+"
        else:
            str_label_orig= "-"
        i+=1
        y.imshow(orig)
        plt.title("pred: "+str_label+" "+"orig: "+str_label_orig)
        y.axes.get_xaxis().set_visible(False)

    plt.savefig(form+"BN_bcancer.pdf",bbox_inches="tight")

def parallel_model(X_train,Y_trainHot,X_test,Y_testHot,class_weight):
    nb_filters =[32,64,64]
    kernel_siz= {}
    kernel_siz[0]= [3,3]
    kernel_siz[1]= [4,4]
    kernel_siz[2]= [5,5]
    input_shape=(50, 50, 3)
    pool_size = (2,2)
    nb_classes =2
    no_parallel_filters = 3
    img_width, img_height = 50, 50
    batch=128
    epochs = 100

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    convs = []
    for k_no in range(no_parallel_filters):
        conv = Conv2D(nb_filters[k_no], kernel_size = (kernel_siz[k_no][0], kernel_siz[k_no][1]),
                    border_mode='same',activation='relu',input_shape=input_shape)(inp)
        pool = MaxPooling2D(pool_size=pool_size)(conv)
        convs.append(pool)

    if len(kernel_siz) > 1:
        out = concatenate(convs)
    else:
        out = convs[0]

    conv_model = Model(input=inp, output=out)

    no_train_samples=len(Y_trainHot)
    no_test_samples=len(Y_testHot)
    tb = TensorBoard(log_dir="logs/{}".format(time()))
    print("beginning first model now")

    model = Sequential()
    model.add(conv_model)
    model.add(Conv2D(32,(3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(32,(3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    #
    # model.add(Conv2D(64,(3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.25))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Conv2D(512,(3,3),activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=True,  # randomly flip images
                                 vertical_flip=False)  # randomly flip images
    model.fit_generator(datagen.flow(X_train, Y_trainHot, batch_size=batch),
                        steps_per_epoch=len(X_train) / 32, epochs=epochs, class_weight=class_weight,verbose=1,
                        validation_data=[X_test, Y_testHot])
                        #callbacks=[MetricsCheckpoint('logs', training_data=(X_train, Y_trainHot), validation_data=(X_test, Y_testHot)), tb])
    score = model.evaluate(X_test, Y_testHot, verbose=1)
    print('parallel model accuracy:', score[1], '\n')
    y_pred = model.predict(X_test)
    auc = roc_auc_score(Y_testHot, y_pred)
    print("auc parallel ",auc)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(Y_testHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')
    Y_pred_classes = np.argmax(y_pred, axis=1)
    # print("y_pred_argmax")
    # print(Y_pred_classes)
    Y_true = np.argmax(Y_testHot, axis=1)
    # print("y_true")
    # print(Y_true)
    correct = (Y_pred_classes == Y_testHot[:, 0]).astype('int32')
    print(correct)
    incorrect = (correct == 0)
    images_error = X_test[incorrect]
    labels_error = Y_pred_classes[incorrect]
    labels_true = Y_testHot[:, 0][incorrect]
    plot_final(X_test,Y_true,class_names,Y_pred_classes,'parallel')
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()),normalize=False, title='Counts',category='parallel')
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()), normalize=True, title='Proportions',category='parallel')
    dfres3 = pd.DataFrame(np.column_stack([auc]), columns=['AUC'])
    dfres3.to_csv("auc_parallel.csv")

def nonParallel_model(X_train,Y_trainHot,X_test,Y_testHot,class_weight):

    batch = 128
    num_classes = 2
    epochs = 100
    tb = TensorBoard(log_dir="logs/{}".format(time()))
    img_rows, img_cols = 50, 50
    input_shape = (img_rows, img_cols, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Conv2D(512,(3,3),activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)
    model.fit_generator(datagen.flow(X_train, Y_trainHot, batch_size=batch),
                        steps_per_epoch=len(X_train) / 32, epochs=epochs, class_weight=class_weight,verbose=1,
                        validation_data=[X_test, Y_testHot])
                       # callbacks=[MetricsCheckpoint('logs', training_data=(X_train, Y_trainHot), validation_data=(X_test, Y_testHot)),tb])
    score = model.evaluate(X_test, Y_testHot, verbose=1)
    print('\n non parallel - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test)
    # print("y_pred")
    # print(y_pred[0:9])
    # print("Y_testhot")
    # print(Y_testHot[0:9])
    auc = roc_auc_score(Y_testHot, y_pred)
    print("auc non parallel ",auc)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(Y_testHot > 0)[1], np.argmax(y_pred, axis=1),
    target_names=list(map_characters.values())), sep='')
    Y_pred_classes = np.argmax(y_pred, axis=1)
    # print("y_pred_argmax")
    # print(Y_pred_classes)
    Y_true = np.argmax(Y_testHot, axis=1)
    # print("y_true")
    # print(Y_true)
    correct = (Y_pred_classes == Y_testHot[:, 0]).astype('int32')
    print(correct)
    incorrect = (correct == 0)
    images_error = X_test[incorrect]
    labels_error = Y_pred_classes[incorrect]
    labels_true = Y_testHot[:, 0][incorrect]
    plot_final(X_test,Y_true,class_names,Y_pred_classes,'nonparallel')
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()),normalize=False, title='Counts',category='nonparallel')
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()), normalize=True, title='Proportions',category='nonparallel')
    dfres3 = pd.DataFrame(np.column_stack([auc]), columns=['AUC'])
    dfres3.to_csv("auc_nonparallel.csv")

def vgg16_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh):
    batch = 128
    num_classes = 2
    epochs = 100
    channel=3
    tb = TensorBoard(log_dir="logs/{}".format(time()))
    img_rows, img_cols = 50, 50
    input_shape = (img_rows, img_cols, channel)
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols,channel)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)
    model.fit_generator(datagen.flow(X_train, Y_trainHot, batch_size=batch),
                        steps_per_epoch=len(X_train) / 32, epochs=epochs, verbose=1,
                        validation_data=[X_test, Y_testHot])
                       # callbacks=[MetricsCheckpoint('logs', training_data=(X_train, Y_trainHot), validation_data=(X_test, Y_testHot)),tb])
    score = model.evaluate(X_test, Y_testHot, verbose=1)
    print('\n vgg16 - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test)
    # print("y_pred")
    # print(y_pred[0:9])
    # print("Y_testhot")
    # print(Y_testHot[0:9])
    auc = roc_auc_score(Y_testHot, y_pred)
    print("auc vgg 16 ",auc)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(Y_testHot > 0)[1], np.argmax(y_pred, axis=1),
    target_names=list(map_characters.values())), sep='')
    Y_pred_classes = np.argmax(y_pred, axis=1)
    # print("y_pred_argmax")
    # print(Y_pred_classes)
    Y_true = np.argmax(Y_testHot, axis=1)
    # print("y_true")
    # print(Y_true)
    correct = (Y_pred_classes == Y_testHot[:, 0]).astype('int32')
    print(correct)
    incorrect = (correct == 0)

    images_error = X_test[incorrect]
    labels_error = Y_pred_classes[incorrect]

    labels_true = Y_testHot[:, 0][incorrect]
    plot_final(X_test,Y_true,class_names,Y_pred_classes,'vgg16')
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()),normalize=False, title='Counts',category='vgg16')
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()), normalize=True, title='Proportions',category='vgg16')
    dfres3 = pd.DataFrame(np.column_stack([auc]), columns=['AUC'])
    dfres3.to_csv("auc_vgg16.csv")

def vgg19_model(X_train, Y_trainHot, X_test, Y_testHot,class_weigh):
    batch = 128
    num_classes = 2
    epochs = 100
    channel=3
    tb = TensorBoard(log_dir="logs/{}".format(time()))
    img_rows, img_cols = 50, 50
    input_shape = (img_rows, img_cols, 3)
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(img_rows, img_cols,channel)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)
    model.fit_generator(datagen.flow(X_train, Y_trainHot, batch_size=batch),
                        steps_per_epoch=len(X_train) / 32, epochs=epochs, verbose=1,
                        validation_data=[X_test, Y_testHot])
                        #callbacks=[MetricsCheckpoint('logs', training_data=(X_train, Y_trainHot), validation_data=(X_test, Y_testHot)),tb])
    score = model.evaluate(X_test, Y_testHot, verbose=1)
    print('\n vgg19 - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test)
    # print("y_pred")
    # print(y_pred[0:9])
    # print("Y_testhot")
    # print(Y_testHot[0:9])
    auc = roc_auc_score(Y_testHot, y_pred)
    print("auc vgg 19 ",auc)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(Y_testHot > 0)[1], np.argmax(y_pred, axis=1),
    target_names=list(map_characters.values())), sep='')
    Y_pred_classes = np.argmax(y_pred, axis=1)
    # print("y_pred_argmax")
    # print(Y_pred_classes)
    Y_true = np.argmax(Y_testHot, axis=1)
    # print("y_true")
    # print(Y_true)
    correct = (Y_pred_classes == Y_testHot[:, 0]).astype('int32')
    print(correct)
    incorrect = (correct == 0)


    images_error = X_test[incorrect]
    labels_error = Y_pred_classes[incorrect]

    labels_true = Y_testHot[:, 0][incorrect]
    plot_final(X_test,Y_true,class_names,Y_pred_classes,'vgg19')
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()),normalize=False, title='Counts',category='vgg19')
    plot_confusion_matrix(cm=confusion_mtx, classes=list(map_characters.values()), normalize=True, title='Proportions',category='vgg19')
    dfres3 = pd.DataFrame(np.column_stack([auc]), columns=['AUC'])
    dfres3.to_csv("auc_vgg19.csv")

if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run classification for breast cancer problem.')
#     parser.add_argument('-p', default=True, help="model having parallel cnn layers stacked together.")
#     parser.add_argument('-np', default=True, help='model having cnn layers placed one after another.')
#     parser.add_argument('-vgg16', default=True, help='Resnet 50 model')
#     parser.add_argument('-vgg19', default=True, help='Resnet 50 model')
#     parser.add_argument('-loc_data',default=os.getcwd(), help="location where data is saved.")
#     parsed_args = parser.parse_args()
    main()

