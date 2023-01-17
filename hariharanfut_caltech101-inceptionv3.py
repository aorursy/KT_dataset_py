import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
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
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
%matplotlib inline
import tensorflow as tf
import tensorflow_datasets as tfdb
import keras
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import scipy
import skimage.transform
import imageio
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = '../input/caltech101/Caltech101/train'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
train = []
train_labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    iter = 0
    for f in os.listdir(path + "/" + category):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = skimage.transform.resize(imageio.imread(fullpath), [128,128, 3])
            img = img.astype('float32')
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            train.append(img) # NORMALIZE IMAGE 
            label_curr = i
            train_labels.append(label_curr)
        #iter = (iter+1)%10;
print ("Num imgs: %d" % (len(train)))
print ("Num labels: %d" % (len(train_labels)) )
print (ncategories)
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = '../input/caltech101/Caltech101/test'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
test = []
test_labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    iter = 0
    for f in os.listdir(path + "/" + category):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = skimage.transform.resize(imageio.imread(fullpath), [128,128, 3])
            img = img.astype('float32')
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            test.append(img) # NORMALIZE IMAGE 
            label_curr = i
            test_labels.append(label_curr)
        #iter = (iter+1)%10;
print ("Num imgs: %d" % (len(test)))
print ("Num labels: %d" % (len(test_labels)) )
print (ncategories)
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = '../input/caltech101/Caltech101/eval'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
vald = []
vald_labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    iter = 0
    for f in os.listdir(path + "/" + category):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = skimage.transform.resize(imageio.imread(fullpath), [128,128, 3])
            img = img.astype('float32')
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            vald.append(img) # NORMALIZE IMAGE 
            label_curr = i
            vald_labels.append(label_curr)
        #iter = (iter+1)%10;
print ("Num imgs: %d" % (len(vald)))
print ("Num labels: %d" % (len(vald_labels)) )
print (ncategories)
train[0].shape
test[0].shape
vald[0].shape
X_train= np.array(train)
X_train= X_train/255.0

X_test= np.array(test)
X_test= X_test/255.0

X_vald= np.array(vald)
X_vald= X_vald/255.0
print(X_test.shape)
print(X_train.shape)
print(X_vald.shape)
X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)
X_vald = np.stack(X_vald, axis=0)
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical( train_labels, num_classes = 102)
y_testHot = to_categorical( test_labels, num_classes = 102)
y_valdHot = to_categorical( vald_labels, num_classes = 102)
print(y_trainHot.shape)
base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(128,128,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(102, activation="softmax")(x)
model_ = Model(inputs=base_model.input, outputs=predictions)

# Lock initial layers to do not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                    , loss='categorical_crossentropy'
                    , metrics=['accuracy'])
model_.summary()
history = model_.fit(X_train, y_trainHot, epochs=200, batch_size=128, validation_data=(X_vald,y_valdHot))
test_loss, test_acc = model_.evaluate(X_test,y_testHot)
print("Test Loss: ", test_loss*100)
print("Test Accuracy: ", test_acc*100)
plt.plot(history.history['accuracy'], 'blue')
plt.plot(history.history['val_accuracy'], 'orange')
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.savefig("Model Accuracy.png",dpi=300,format="png")

print("VGG -Validation Loss: ", history.history['val_loss'][-1]*100)
print("VGG - Validation Accuracy: ", history.history['val_accuracy'][-1]*100)
print("\n")
print("VGG - Training Loss: ", history.history['loss'][-1]*100)
print("VGG - Training Accuracy: ", history.history['accuracy'][-1]*100)
print("\n")
plt.plot(history.history['loss'], 'blue')
plt.plot(history.history['val_loss'], 'orange')
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.savefig("Model Loss.png",dpi=300,format="png")
target_names = []
for i in range(101):
    a = 'Object '
    b = str(i)
    c = a+b
    c = [i]
    target_names.append((a+b))

def reports(X_test,y_test):
    Y_pred = model_.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model_.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0]*100
    Test_accuracy = score[1]*100
    kc=cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    #mse=mean_squared_error(y_test, y_pred)
    #mae=mean_absolute_error(y_test, y_pred)
    #precision=precision_score(y_test, y_pred, average='weighted')
    #print(classification_report(y_test, y_pred, target_names=target_names))


    
    return classification, confusion, Test_Loss, Test_accuracy ,kc#,mse,mae
from sklearn.metrics import classification_report, confusion_matrix,cohen_kappa_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score
# calculate result, loss, accuray and confusion matrix
classification, confusion, Test_loss, Test_accuracy,kc = reports(X_test,y_testHot)
classification = str(classification)
confusion_str = str(confusion)
print("confusion matrix: ")
print('{}'.format(confusion_str))
print("KAppa Coeefecient=",kc)
print('Test loss {} (%)'.format(Test_loss))
print('Test accuracy {} (%)'.format(Test_accuracy))
#print("Mean Squared error=",mse)
#print("Mean absolute error=",mae)
print(classification)
import matplotlib.pyplot as plt
%matplotlib inline
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = Normalized*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure(figsize=(100,100))
plot_confusion_matrix(confusion, classes=target_names, normalize=False, 
                      title='Confusion matrix, without normalization')
plt.savefig("confusion_matrix_without_normalization.png")
plt.show()
plt.figure(figsize=(100,100))
plot_confusion_matrix(confusion, classes=target_names, normalize=True, 
                      title='Normalized confusion matrix')
plt.savefig("confusion_matrix_with_normalization.png")
plt.show()
model_.save_weights('caltech101-inceptionv3.h5')