# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv # image processing

import matplotlib.pyplot as plt # plot images



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Global Parameters

RESHAPE_SIZE = 100

RANDOM_STATE = 420

MAX_ITER = 1000

ERR_TOL = 0.01
"""

Consultation taken from:

https://stackoverflow.com/questions/41907598/how-to-train-images-when-they-have-different-size

"""



def load_resize_images_from_folder(folder):

    """

    Function to load and resize the contents of a folder into a list of numpy arrays

    Function adopted from:

    https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder

    """

    images = []

    for filename in os.listdir(folder):

        img = cv.imread(os.path.join(folder,filename))

        if img is not None:

            img = cv.resize(img, (RESHAPE_SIZE, RESHAPE_SIZE)).flatten()

            images.append(img)

    return images



def shuffle_index(x, y):

    """Shuffle the index of train or test data in order to ensure random cross-val folds"""

    shuffle_index = np.random.permutation(len(x))

    x, y = x[shuffle_index], y[shuffle_index]

    return x, y



def label_y(len_x, len_train_cactus):

    """Make a list of labels to be used in training based on training indexes"""

    y = []

    for elem in range(len_x):

        if elem <= len_train_cactus:

            y.append("cact")

        else:

            y.append("no_cact")

    return np.array([1 if elem == 'cact' else 0 for elem in y])
#Returns lists of numpy arrays of shape depending on image size

train_cactus = load_resize_images_from_folder("/kaggle/input/training_set/training_set/cactus")

train_no_cactus = load_resize_images_from_folder("/kaggle/input/training_set/training_set/no_cactus")

test_cactus = load_resize_images_from_folder("/kaggle/input/validation_set/validation_set/cactus")

test_no_cactus = load_resize_images_from_folder("/kaggle/input/validation_set/validation_set/no_cactus")



x_train = list()

x_train += train_cactus

x_train += train_no_cactus

x_train = np.array(x_train)



y_train = label_y(len(x_train), len(train_cactus))



x_train, y_train = shuffle_index(x_train, y_train)



x_test = list()

x_test += test_cactus

x_test += test_no_cactus

x_test = np.array(x_test)



y_test = label_y(len(x_test), len(test_cactus))



x_test, y_test = shuffle_index(x_test, y_test)
from random import randint



def plot_examples_RGB(cact_ind, no_cact_ind):

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    ax1.set_title('Cactus')

    ax1.imshow(train_cactus[cact_ind].reshape(RESHAPE_SIZE, RESHAPE_SIZE, 3))

    ax2.set_title('Not Cactus')

    ax2.imshow(train_no_cactus[no_cact_ind].reshape(RESHAPE_SIZE, RESHAPE_SIZE, 3))

    

plot_examples_RGB(

    randint(0,len(train_cactus)), 

    randint(0,len(train_no_cactus)))
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=MAX_ITER, tol=ERR_TOL, random_state=RANDOM_STATE)

sgd_clf.fit(x_train, y_train) 
from sklearn.model_selection import cross_val_score



training_score = cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")

test_score = cross_val_score(sgd_clf, x_test, y_test, cv=3, scoring="accuracy")



print("Training Score: "+str(sum(training_score)/len(training_score)))

print("Cross-Validation Score: "+str(sum(test_score)/len(test_score)))
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



#Display confusion matrix

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)

confusion_matrix = confusion_matrix(y_train, y_train_pred)

print(confusion_matrix)
from sklearn.externals import joblib



# Dump model

filename = 'rgb_model_v1.sav'

joblib.dump(sgd_clf, filename)