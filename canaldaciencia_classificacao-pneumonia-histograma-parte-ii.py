# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
import pickle

from os import listdir

from os.path import isfile, join

import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
clf = pickle.load(open('/kaggle/input/decisiontreepneumonia/pneumonia_histograma (1).sav', 'rb'))
mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'



normal_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
histogramas_normais=[]

mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'



y_normal=[]



for file_path in (normal_files[:200]):

    image =cv2.imread(mypath+file_path)

    histogram=cv2. cv2.calcHist([image],      # image

                               [0, 1],           # channels

                               None,             # no mask

                               [180, 256],       # size of histogram

                               [0, 180, 0, 256]  # channel values

                               )



    histogramas_normais.append( histogram.ravel())

    y_normal.append(0)


mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'



pneumonia_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
histogramas_pneumonia=[]

mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'





y_pneumonia=[]



for file_path in (pneumonia_files[:200]):

    image =cv2.imread(mypath+file_path)

    histogram=cv2. cv2.calcHist([image],      # image

                               [0, 1],           # channels

                               None,             # no mask

                               [180, 256],       # size of histogram

                               [0, 180, 0, 256]  # channel values

                               )



    histogramas_pneumonia.append( histogram.ravel())

    y_pneumonia.append(1)

y=y_normal+y_pneumonia
X=histogramas_normais+histogramas_pneumonia
df=pd.DataFrame(X)
df['y']=y
X=df.drop(['y'],axis=1)
y_test=df['y']
y_pred=clf.predict(X)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)




def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):



    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
from sklearn.metrics import confusion_matrix

c=confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm           = c, 

                      normalize    = False,

                      target_names = ['Normal', 'Pneumonia'],

                      title        = "Confusion Matrix")
from sklearn import metrics

y = np.array([1, 1, 2, 2])

scores = np.array([0.1, 0.4, 0.35, 0.8])

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

# generate 2 class dataset

plt.figure(figsize=(10,10))

y_valid=y_test

# predict probabilities

lr_probs = y_pred

# keep probabilities for the positive outcome only

lr_probs = lr_probs

# calculate scores

ns_auc = roc_auc_score(y_valid, y_pred)

lr_auc = roc_auc_score(y_valid, lr_probs)

# summarize scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))

print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_valid, y_pred)

lr_fpr, lr_tpr, _ = roc_curve(y_valid, lr_probs)

# plot the roc curve for the model

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

# show the legend

pyplot.legend()

# show the plot

pyplot.show()
from sklearn.metrics import multilabel_confusion_matrix

mcm = multilabel_confusion_matrix(y_test, y_pred)

tn = mcm[:, 0, 0]

tp = mcm[:, 1, 1]

fn = mcm[:, 1, 0]

fp = mcm[:, 0, 1]
tp / (tp + fn)
tn / (tn + fp)
from itertools import product

from sklearn.tree import DecisionTreeClassifier





def plot_area(X,y,clf,name):

    plt.figure(figsize=(10,10))

   

    clf.fit(X.iloc[:, 0:2],y)

    # Plotting decision regions

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1

    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                         np.arange(y_min, y_max, 0.1))



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z, alpha=0.4)



    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y,

                                  s=20, edgecolor='k')



        

plt.show()


clf=DecisionTreeClassifier()

plot_area(X,y_test,clf,'Decision Tree')