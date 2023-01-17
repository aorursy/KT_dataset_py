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
import pandas as pd
import numpy as np
from os import listdir
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from scipy.misc import imresize, imread
from scipy import misc
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
%matplotlib inline
train_path = '../input/train/'
test_path = '../input/test/'
listdir(train_path)
maize_path= train_path + 'Maize/'
listdir(maize_path)[:10]
image_path = maize_path + 'a5c2eec2d.png'
image = cv2.imread(image_path)
plt.figure(figsize=(16,16))
plt.imshow(image)
image.shape
path_name = train_path + '/**/*.png'
imagePatches = glob(path_name, recursive=True)
for filename in imagePatches[0:10]:
    print(filename)
# Plot Multiple Images
bunchOfImages = imagePatches
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in bunchOfImages[:25]:
    im = cv2.imread(l)
    im = cv2.resize(im, (50, 50)) 
    plt.subplot(5, 5, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tqdm import tqdm
imagesize = 200
def loadBatchImages(path):
    catList = listdir(path)
    loadedImagesTrain = []
    loadedLabelsTrain = []
    for cat in catList:
        deepPath = path+cat+"/"
        imageList = listdir(deepPath)
        for images in tqdm(imageList):                
            img = load_img(deepPath + images)
            img = misc.imresize(img, (imagesize,imagesize))
            img = img_to_array(img)
            loadedLabelsTrain.append(cat)
            loadedImagesTrain.append(img)
    return loadedImagesTrain, loadedLabelsTrain
loadedImagesTrain, loadedLabelsTrain = loadBatchImages(train_path)
#Encode labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
loadedLabelsTrain = np.asarray(loadedLabelsTrain)
encoder.fit(loadedLabelsTrain)
encoded_loadedLabelsTrain = encoder.transform(loadedLabelsTrain)
num_classes =len(np.unique(loadedLabelsTrain))
num_classes
del loadedLabelsTrain
import gc
gc.collect()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(loadedImagesTrain,
                                                    encoded_loadedLabelsTrain,
                                                    test_size=0.2) 
del loadedImagesTrain,encoded_loadedLabelsTrain
gc.collect()
X_train=np.array(X_train)
X_train=X_train/255.0

X_test=np.array(X_test)
X_test=X_test/255.0
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
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = num_classes)
y_testHot = to_categorical(y_test, num_classes = num_classes)
imageSize = 200
def createNetwork(pretrained_model,imageSize):
    base_model = pretrained_model # Topless
    optimizer1 = keras.optimizers.Adam()
    # Add top layer
    x = base_model.output
    x = Conv2D(imageSize, kernel_size = (3,3), padding = 'valid')(x)
    x = Flatten()(x)
    x = Dropout(0.75)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer1, 
                  metrics=['accuracy'])
    model.summary()
    return model

from keras.applications import VGG16,InceptionV3
from keras.applications.vgg16 import preprocess_input
pretrained_model_VGG16 = VGG16(include_top=False, input_shape=(imageSize, imageSize, 3))
model = createNetwork(pretrained_model_VGG16,imageSize)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
a = X_train
b = y_trainHot
c = X_test
d = y_testHot
epochs = 10
batch_size = 32
history = model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=len(a) / batch_size, 
                              epochs=epochs,validation_data = [c, d],
                              callbacks = [MetricsCheckpoint('logs')])
imageList = listdir(test_path)
loadedImages = []
for images in tqdm(imageList):                
    img = load_img(test_path + images)
    img = misc.imresize(img, (imagesize,imagesize))
    img = img_to_array(img)
    loadedImages.append(img)

X_test=np.array(loadedImages)
X_test=X_test/255.0
y_pred = model.predict(X_test)
y_pred[:5]
Y_pred_classes = np.argmax(y_pred,axis=1) 
Y_pred_classes[:5]
predictions = encoder.inverse_transform(Y_pred_classes)
print("Size of predictions ",predictions.shape[0])
print("Size of Test ",X_test.shape[0])
predictions[:5]
pretrained_model_InceptionV3 = InceptionV3(include_top=False, input_shape=(imageSize, imageSize, 3))
base_model = pretrained_model_InceptionV3 # Topless
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Train top layer
for layer in base_model.layers:
    layer.trainable = False

optimizer1 = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer1, 
              metrics=['accuracy'])
history = model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=len(a) / batch_size, 
                              epochs=epochs,validation_data = [c, d],
                              callbacks = [MetricsCheckpoint('logs')])
