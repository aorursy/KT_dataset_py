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
import numpy as np

import tensorflow 



import keras

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.optimizers import Adam

from keras.optimizers import RMSprop

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.preprocessing import image

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

from keras.models import load_model



import matplotlib.pyplot as plt

import itertools

from matplotlib import pyplot as plt

import glob

import matplotlib.image as mpimg

from keras.preprocessing import image

import imageio as im

import glob

import matplotlib



pwd
test_path = '/kaggle/input/plantvillagepredictions/val/val'
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',

                                                                                                    'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',

                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',

                                                                                                    'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10,shuffle=False)
from keras.models import load_model
model = Sequential()
pwd
model = load_model('/kaggle/input/finalmodel/NewOne5084acc993v121.h5')
predicted_label = model.predict_generator(test_batches,steps=440,verbose=0)
true_labels = test_batches.classes
labels = np.zeros((4396,1))
count = 0

for i in range(0,4396):

    for j in range(0,38):

        if predicted_label[i][j] == max(predicted_label[i]):

            labels[i] = j

            if j == true_labels[i]:

                count+=1

print(count)

print(count/4396)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# No Batch-Normalization and No Dropout + Augmentation "CustomModel.h5"

print(f1_score(labels,true_labels, average="macro"))

print(precision_score(labels,true_labels, average="macro"))

print(recall_score(labels,true_labels, average="macro"))  
model.summary()
import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels





def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    print(classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
class_names = ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])



class1 = np.array((class_names),dtype=np.int)

type(class1[1]),len(class_names)
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

# Plot non-normalized confusion matrix

plot_confusion_matrix(true_labels,labels, classes=class1,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(true_labels,labels, classes=class1, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
from sklearn.metrics import classification_report
target_names =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20','c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30','c_31','c_32','c_33','c_34','c_35','c_36','c_37'],
print(len(target_names[0]))
print(classification_report(true_labels, labels, target_names=target_names[0]))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline
true = sns.countplot(true_labels)

true_labels.shape
labels1 = labels.reshape(4396,)
preds = sns.countplot(labels1)
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap='binary',aspect='auto')

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

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



# Predict the values from the validation dataset



# compute the confusion matrix

confusion_mtx = confusion_matrix(true_labels,labels1) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(38))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix



def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):

    """

    Generate matrix plot of confusion matrix with pretty annotations.

    The plot image is saved to disk.

    args: 

      y_true:    true label of the data, with shape (nsamples,)

      y_pred:    prediction of the data, with shape (nsamples,)

      filename:  filename of figure file to save

      labels:    string array, name the order of class labels in the confusion matrix.

                 use `clf.classes_` if using scikit-learn models.

                 with shape (nclass,).

      ymap:      dict: any -> string, length == nclass.

                 if not None, map the labels & ys to more understandable strings.

                 Caution: original y_true, y_pred and labels must align.

      figsize:   the size of the figure plotted.

    """

    if ymap is not None:

        y_pred = [ymap[yi] for yi in y_pred]

        y_true = [ymap[yi] for yi in y_true]

        labels = [ymap[yi] for yi in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=labels, columns=labels)

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    plt.savefig(filename)
cm_analysis(true_labels, labels1, 'cm.png', class1, ymap=None, figsize=(20,20))