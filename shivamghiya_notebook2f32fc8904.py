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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the necessary packages and libaries

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn import svm, datasets

import matplotlib.pyplot as plt
# import some data to play with

iris = datasets.load_iris()

# we only take the first two features (1 and 2)

Xa = iris.data[:, :2]

y = iris.target

Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, y, train_size=0.8, random_state = 0)
# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

# SVM regularization parameter C = 1.0 

# Kernel can be {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Xa_train, ya_train)

rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(Xa_train, ya_train)

poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(Xa_train, ya_train)



#create the mesh

x_min, x_max = Xa[:, 0].min() - 1, Xa[:, 0].max() + 1

y_min, y_max = Xa[:, 1].min() - 1, Xa[:, 1].max() + 1

h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))



# create the title that will be shown on the plot

titles = ['Linear kernel','RBF kernel','Polynomial kernel']

for i, clf in enumerate((linear, rbf, poly)):

    #defines how many plots: 2 rows, 2columns=> leading to 4 plots

    plt.subplot(2, 2, i + 1) #i+1 is the index

    #space between plots

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)

    # Plot also the training points

    plt.scatter(Xa[:, 0], Xa[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.xticks(())

    plt.yticks(())

    plt.title(titles[i])

plt.show()
linear_pred = linear.predict(Xa_test)

poly_pred = poly.predict(Xa_test)

rbf_pred = rbf.predict(Xa_test)
# retrieve the accuracy and print it for all 4 kernel functions

accuracy_lin = linear.score(Xa_test, ya_test)

accuracy_poly = poly.score(Xa_test, ya_test)

accuracy_rbf = rbf.score(Xa_test, ya_test)

print('Accuracy Linear Kernel:', accuracy_lin)

print('Accuracy Polynomial Kernel:', accuracy_poly)

print('Accuracy Radial Basis Kernel:', accuracy_rbf)
# creating a confusion matrix

cm_lin = confusion_matrix(ya_test, linear_pred)

cm_poly = confusion_matrix(ya_test, poly_pred)

cm_rbf = confusion_matrix(ya_test, rbf_pred)

print(cm_lin)

print(cm_poly)

print(cm_rbf)
# get support vectors

linear_sv = linear.support_vectors_

rbf_sv = rbf.support_vectors_

poly_sv = poly.support_vectors_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)
# get indices of support vectors

linear_sv = linear.support_

rbf_sv = rbf.support_

poly_sv = poly.support_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)
# get indices of support vectors

linear_sv = linear.n_support_

rbf_sv = rbf.n_support_

poly_sv = poly.n_support_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)
# we only take the last two features (3 and 4)

Xb = iris.data[:, 2:]

y = iris.target

Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, y, train_size=0.8, random_state = 0)
# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

# SVM regularization parameter C = 1.0 

# Kernel can be {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Xb_train, yb_train)

rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(Xb_train, yb_train)

poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(Xb_train, yb_train)



#create the mesh

x_min, x_max = Xb[:, 0].min() - 1, Xb[:, 0].max() + 1

y_min, y_max = Xb[:, 1].min() - 1, Xb[:, 1].max() + 1

h = .01

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))



# create the title that will be shown on the plot

titles = ['Linear kernel','RBF kernel','Polynomial kernel']

for i, clf in enumerate((linear, rbf, poly)):

    #defines how many plots: 2 rows, 2columns=> leading to 4 plots

    plt.subplot(2, 2, i + 1) #i+1 is the index

    #space between plots

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)

    # Plot also the training points

    plt.scatter(Xb[:, 0], Xb[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.xticks(())

    plt.yticks(())

    plt.title(titles[i])

plt.show()
linear_pred = linear.predict(Xb_test)

poly_pred = poly.predict(Xb_test)

rbf_pred = rbf.predict(Xb_test)
# retrieve the accuracy and print it for all 4 kernel functions

accuracy_lin = linear.score(Xb_test, yb_test)

accuracy_poly = poly.score(Xb_test, yb_test)

accuracy_rbf = rbf.score(Xb_test, yb_test)

print('Accuracy Linear Kernel:', accuracy_lin)

print('Accuracy Polynomial Kernel:', accuracy_poly)

print('Accuracy Radial Basis Kernel:', accuracy_rbf)
# creating a confusion matrix

cm_lin = confusion_matrix(yb_test, linear_pred)

cm_poly = confusion_matrix(yb_test, poly_pred)

cm_rbf = confusion_matrix(yb_test, rbf_pred)

print(cm_lin)

print(cm_poly)

print(cm_rbf)
# get support vectors

linear_sv = linear.support_vectors_

rbf_sv = rbf.support_vectors_

poly_sv = poly.support_vectors_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)
# get indices of support vectors

linear_sv = linear.support_

rbf_sv = rbf.support_

poly_sv = poly.support_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)
# get indices of support vectors

linear_sv = linear.n_support_

rbf_sv = rbf.n_support_

poly_sv = poly.n_support_

print('------------------------------------------------------------\n',linear_sv)

print('------------------------------------------------------------\n',rbf_sv)

print('------------------------------------------------------------\n',poly_sv)