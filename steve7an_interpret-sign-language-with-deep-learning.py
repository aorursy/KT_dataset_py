import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import models, layers, optimizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.utils import class_weight

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

from keras.models import Sequential, model_from_json

from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

import os

from glob import glob

import matplotlib.pyplot as plt

import random

import cv2

import pandas as pd

import numpy as np

import matplotlib.gridspec as gridspec

import seaborn as sns

import zlib

import itertools

import sklearn

import itertools

import scipy

import skimage

from skimage.transform import resize

import csv

from tqdm import tqdm

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

#from keras.applications.mobilenet import MobileNet

#from sklearn.metrics import roc_auc_score

#from sklearn.metrics import roc_curve

#from sklearn.metrics import auc

#import warnings

#warnings.filterwarnings("ignore")

%matplotlib inline
import dask

import dask.dataframe as dd

import dask.array as da
# print(os.listdir("../input"))

# print(os.listdir("../input/asl-alphabet"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A"))
imageSize=50

train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

test_dir =  "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"

from tqdm import tqdm

def get_data(folder):

    """

    Load the data and labels from the given folder.

    """

    X = []

    y = []

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['A']:

                label = 0

            elif folderName in ['B']:

                label = 1

            elif folderName in ['C']:

                label = 2

            elif folderName in ['D']:

                label = 3

            elif folderName in ['E']:

                label = 4

            elif folderName in ['F']:

                label = 5

            elif folderName in ['G']:

                label = 6

            elif folderName in ['H']:

                label = 7

            elif folderName in ['I']:

                label = 8

            elif folderName in ['J']:

                label = 9

            elif folderName in ['K']:

                label = 10

            elif folderName in ['L']:

                label = 11

            elif folderName in ['M']:

                label = 12

            elif folderName in ['N']:

                label = 13

            elif folderName in ['O']:

                label = 14

            elif folderName in ['P']:

                label = 15

            elif folderName in ['Q']:

                label = 16

            elif folderName in ['R']:

                label = 17

            elif folderName in ['S']:

                label = 18

            elif folderName in ['T']:

                label = 19

            elif folderName in ['U']:

                label = 20

            elif folderName in ['V']:

                label = 21

            elif folderName in ['W']:

                label = 22

            elif folderName in ['X']:

                label = 23

            elif folderName in ['Y']:

                label = 24

            elif folderName in ['Z']:

                label = 25

            elif folderName in ['del']:

                label = 26

            elif folderName in ['nothing']:

                label = 27

            elif folderName in ['space']:

                label = 28           

            else:

                label = 29

            for image_filename in tqdm(os.listdir(folder + folderName)):

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = da.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

           # break

    X = da.asarray(X) #.replace(0, np.nan).to_sparse()

    y = da.asarray(y) #.replace(0, np.nan).to_sparse()

    return X,y

X_train, y_train = get_data(train_dir) 

#X_test, y_test= get_data(test_dir) # Too few images



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 



# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(y_train, num_classes = 30)

y_testHot = to_categorical(y_test, num_classes = 30)
# Shuffle data to permit further subsampling

from sklearn.utils import shuffle

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)

X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)

X_train = X_train[:30000]

X_test = X_test[:30000]

y_trainHot = y_trainHot[:30000]

y_testHot = y_testHot[:30000]
X_train.shape
type(X_train)
import h5py    

f = h5py.File('X_train.hdf5')

d = f.require_dataset('/data', shape=X_train.shape, dtype=X_train.dtype)

da.store(X_train, d)
f = h5py.File('X_test.hdf5')

d = f.require_dataset('/data', shape=X_test.shape, dtype=X_test.dtype)

da.store(X_test, d)
np.savetxt('y_trainHot.txt', y_trainHot)
np.savetxt('y_testHot.txt', y_testHot)


f = h5py.File('X_train.hdf5')

d = f.require_dataset('/data', shape=X_train.shape, dtype=X_train.dtype)

da.store(X_train, d)
import numpy as np

np.savetxt("X_train.csv", X_train, delimiter=",")

X_train.to_csv('X_train.csv',index=False)

X_test.to_csv('X_test.csv',index=False)

y_trainHot.to_csv('y_trainHot.csv',index=False)

y_testHot.to_csv('y_testHot.csv',index=False)

def plotHistogram(a):

    """

    Plot histogram of RGB Pixel Intensities

    """

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(a)

    plt.axis('off')

    histo = plt.subplot(1,2,2)

    histo.set_ylabel('Count')

    histo.set_xlabel('Pixel Intensity')

    n_bins = 30

    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

plotHistogram(X_train[1])
multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/**')

def plotThreeImages(images):

    r = random.sample(images, 3)

    plt.figure(figsize=(16,16))

    plt.subplot(131)

    plt.imshow(cv2.imread(r[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(r[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(r[2]))

    #;

plotThreeImages(multipleImages)
multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/B/**')

def plotThreeImages(images):

    r = random.sample(images, 3)

    plt.figure(figsize=(16,16))

    plt.subplot(131)

    plt.imshow(cv2.imread(r[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(r[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(r[2])) 

plotThreeImages(multipleImages)
print("A")

multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
print("B")

multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/B/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

dict_characters=map_characters

import seaborn as sns

df = pd.DataFrame()

df["labels"]=y_train.compute()

lab = df['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)
# Helper Functions  Learning Curves and Confusion Matrix

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



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

    plt.ylabel('Accuracy')



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (8,8))

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

    plt.subplot(1,2,2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./loss_curve.png')
map_characters1 = map_characters

class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train.compute()), y_train.compute())

weight_path1 = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

weight_path2 = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model_1 = VGG16(weights = weight_path1, include_top=False, input_shape=(imageSize, imageSize, 3))

#pretrained_model_2 = InceptionV3(weights = weight_path2, include_top=False, input_shape=(imageSize, imageSize, 3))

optimizer1 = keras.optimizers.Adam()

optimizer2 = keras.optimizers.RMSprop(lr=0.0001)

def pretrainedNetwork(xtrain,ytrain,xtest,ytest,pretrainedmodel,pretrainedweights,classweight,numclasses,numepochs,optimizer,labels):

    base_model = pretrained_model_1 # Topless

    # Add top layer

    x = base_model.output

    x = Flatten()(x)

    predictions = Dense(numclasses, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train top layer

    for layer in base_model.layers:

        layer.trainable = False

    model.compile(loss='categorical_crossentropy', 

                  optimizer=optimizer, 

                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    model.summary()

    # Fit model

    history = model.fit(xtrain,ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])

    # Evaluate model

    score = model.evaluate(xtest,ytest, verbose=0)

    print('\nKeras CNN - accuracy:', score[1], '\n')

    y_pred = model.predict(xtest)

    print('\n', sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='') 

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(ytest,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plotKerasLearningCurve()

    plt.show()

    plot_learning_curve(history)

    plt.show()

    plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))

    plt.show()

    return model

pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,30,10,optimizer1,map_characters1)
print('\n', sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='') 

Y_pred_classes = np.argmax(y_pred,axis = 1) 

Y_true = np.argmax(ytest,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plotKerasLearningCurve()

plt.show()

plot_learning_curve(history)

plt.show()

plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))

plt.show()
# To Do: (1) try using more than 30000 of the 87000 images; (2) try using larger images; (3) try using different network architectures 