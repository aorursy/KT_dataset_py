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
# set up constants

RANDOM_STATE = 42
# loading data

df_full = pd.read_csv('../input/Iris.csv')

df_full.head()
df_full.info()
df_full['Target'], labels = pd.factorize(df_full['Species'])

df_full.head()
df_full_corr = df_full.corr()

df_full_corr.style.background_gradient()
from sklearn.model_selection import train_test_split 



x = df_full.drop(['Species', 'Target'], axis=1)

y = df_full['Target']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RANDOM_STATE)
from sklearn import svm

from sklearn.metrics import confusion_matrix 



clf = svm.SVC(gamma='scale', decision_function_shape='ovo')

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)  

y_pred = clf.predict(x_test)  # predicted values

cm = confusion_matrix(y_test, y_pred)

print("Accuracy = {}".format(accuracy))
from matplotlib import pyplot as plt

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

plot_confusion_matrix(cm, labels, True)