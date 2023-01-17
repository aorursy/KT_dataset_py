# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt





def make_meshgrid(ax, h=.02):

    # x_min, x_max = x.min() - 1, x.max() + 1

    # y_min, y_max = y.min() - 1, y.max() + 1

    x_min, x_max = ax.get_xlim()

    y_min, y_max = ax.get_ylim()



    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out





def draw_boundary(ax, clf):



    xx, yy = make_meshgrid(ax)

    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



def plot_SVM(gamma=1, C=1):

    aaron_judge = pd.read_csv('../input/aaron_judge.csv')

    #print(aaron_judge.columns)

    #print(aaron_judge.description.unique())

    #print(aaron_judge.type.unique())

    aaron_judge.type = aaron_judge.type.map({'S':1, 'B':0})

    #print(aaron_judge.type)

    #print(aaron_judge['plate_x'])

    aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])

    fig, ax = plt.subplots()

    plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap = plt.cm.coolwarm, alpha=0.6)

    training_set, validation_set = train_test_split(aaron_judge, random_state=1)

    classifier = SVC(kernel='rbf', gamma = gamma, C = C)

    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

    draw_boundary(ax, classifier)

    plt.show()

    print("The score of SVM with gamma={0} and C={1} is:".format(gamma, C) )

    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

# likely underfitting

plot_SVM(gamma=0.03, C=0.05)
# overfitting

plot_SVM(gamma=100, C=100)
# good choice

plot_SVM(gamma=3, C=1)