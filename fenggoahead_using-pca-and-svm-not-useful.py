# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

import collections

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def data_processing(N,tile_size):

    master = pd.read_csv("../input/train_labels.csv")

    img_path = "../input/train/"

    train_x,train_y = [],[]

    file_path = []

    if N == -1 : N = len(master)

    for i in range(N):

        file_path = img_path + str(master.ix[i][0]) +'.jpg' 

        img = cv2.imread(file_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_resize = cv2.resize(img, dsize=tile_size)

        train_y.append(master.ix[i][1])

        train_x.append(img_resize)

        

        train_y.append(master.ix[i][1])

        img_flip_resize = cv2.flip(img_resize,1)

        train_x.append(img_flip_resize)

                

    train_x = np.array(train_x)

    

    s1,s2,s3 = np.shape(train_x)

    train_x = np.reshape(train_x,[s1,s2*s3])

    train_y = np.array(train_y)

    return train_x,train_y
N = -1

tile_size=(128,128*886//1154)

x_train,y_train = data_processing(N,tile_size)

print('y_distribution:',collections.Counter(y_train) )

N = len(y_train)

x1_train,x1_cross,y1_train,y1_cross = train_test_split(x_train[:N], y_train[:N], 

                                                       train_size=0.7, random_state=1)
def plot_img(img_array,tile_size):

    img_array = np.reshape(img_array,[tile_size[1],tile_size[0]])

    plt.imshow(img_array)

    plt.show()
N_array=[32]

Acc_train_arr,Acc_cross_arr = [],[]

print('size of image:',tile_size)

print('N:',N)

for COMPONENT_NUM in N_array:

     pca = PCA(n_components=COMPONENT_NUM, whiten=True)

     pca.fit(x1_train)

     x2_train = pca.transform(x1_train)

     y2_train = y1_train

     x2_cross = pca.transform(x1_cross)

     y2_cross = y1_cross

     SVM_classifier = SVC(C=1,gamma=0.1)

     SVM_classifier.fit(x2_train,y2_train)

     predicted = SVM_classifier.predict(x2_cross)

     Acc_train = SVM_classifier.score(x2_train, y2_train)

     Acc_cross = SVM_classifier.score(x2_cross, y2_cross)              

     print("NN= %3f, Acc_train=%3f, Acc_cross = %3f "% (COMPONENT_NUM,Acc_train,Acc_cross ))
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
predicted = SVM_classifier.predict(x2_cross) 

n_view = 4

cnf_matrix = confusion_matrix(y2_cross, predicted)

np.set_printoptions(precision=2)

context = ['harmless:', 'invasive']

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=context,normalize=False,

                      title='Confusion matrix, without normalization')

plt.show()