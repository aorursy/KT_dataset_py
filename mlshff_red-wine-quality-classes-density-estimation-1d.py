# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Load libraries

import pandas

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.preprocessing import label_binarize

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from itertools import cycle

from sklearn.metrics import roc_curve, auc

from scipy import interp





from sklearn import mixture

from scipy.stats import multivariate_normal

from matplotlib.colors import LogNorm



from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D



from sklearn.neighbors import KernelDensity



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read Iris.csv

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 

         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

dataset = pandas.read_csv('../input/winequality-red.csv', names=names)



#show dataset info

print(dataset.describe())

print('')

print(dataset.groupby('quality').size())
array = dataset.values

X = array[:,0:1]

Y = array[:,11]



for i in range(0, Y.size):

    if Y[i] >= 7:

        Y[i] = 0 #GOOD WINE

    else:

        Y[i] = 1 #BAD WINE

        

test_size = 0.20

seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 

                                                                                test_size=test_size, 

                                                                                random_state=seed)



# Test options and evaluation metric

scoring = 'f1'
X_train_c0 = []

X_train_c1 = []



for i in range(0, X_train[:,0:1].size):

    if Y_train[i] == 0:

        X_train_c0.append(X_train[i])

    else:

        if Y_train[i] == 1:

            X_train_c1.append(X_train[i])

        else:

            X_train_c2.append(X_train[i])

        

X_train_0 = np.array(X_train_c0)

X_train_1 = np.array(X_train_c1)
X_train_0_mean = X_train_0[:,0:1].mean()#[X_train_0[:,0:1].mean(), X_train_0[:,1:2].mean()]

print('X_train mean of class 0 ' + str(X_train_0_mean))

X_train_0_var = X_train_0[:,0:1].var()#[X_train_0[:,0:1].var(), X_train_0[:,1:2].var()]

#print('X_train var of class 0 ' + str(X_train_0_var))



X_train_1_mean = X_train_1[:,0:1].mean()#[X_train_1[:,0:1].mean(), X_train_1[:,1:2].mean()]

print('X_train mean of class 1 ' + str(X_train_1_mean))

X_train_1_var = X_train_1[:,0:1].var()#[X_train_1[:,0:1].var(), X_train_1[:,1:2].var()]

#print('X_train var of class 1 ' + str(X_train_1_var))



df_0 = pandas.DataFrame(X_train_0)



df_1 = pandas.DataFrame(X_train_1)
X_train_min = X_train_0[:,0:1].min()#[X_train[:,0:1].min(), X_train[:,1:2].min()]

X_train_max = X_train[:,0:1].max()#[X_train[:,0:1].max(), X_train[:,1:2].max()]



print('X_train min val ' + str(X_train_min))

print('X_train max val ' + str(X_train_max))
xls0 = np.linspace(X_train_min, X_train_max)



import math
Z_0 = []

for i in range(0, xls0.size):

    s = (xls0[i] - X_train_0_mean)

    s *= s

    s /= 2 * X_train_0_var

    s *= -1

    s = math.exp(s)

    s = s / math.sqrt(2 * 3.1415 * X_train_0_var)

    Z_0.append(s)



plt.plot(xls0, Z_0)
Z_1 = []

for i in range(0, xls0.size):

    s = (xls0[i] - X_train_1_mean)

    s *= s

    s /= 2 * X_train_1_var

    s *= -1

    s = math.exp(s)

    s = s / math.sqrt(2 * 3.1415 * X_train_1_var)

    Z_1.append(s)



plt.plot(xls0, Z_1)
kde_0 = KernelDensity(bandwidth=1, kernel='epanechnikov')

kde_0.fit(X_train_0, 0)



x0 = np.array(xls0)

Z_0 = np.exp(kde_0.score_samples(x0.reshape(-1,1)))



plt.plot(xls0, Z_0)
kde_1 = KernelDensity(bandwidth=1, kernel='epanechnikov')

kde_1.fit(X_train_1, 1)



Z_1 = np.exp(kde_1.score_samples(x0.reshape(-1,1)))



plt.plot(xls0, Z_1)
gm_0 = mixture.GaussianMixture(n_components=3, covariance_type='full')

gm_0.fit(X_train_0)

    

Z_0 = np.exp(gm_0.score_samples(x0.reshape(-1,1)))



plt.plot(xls0, Z_0)
gm_1 = mixture.GaussianMixture(n_components=3, covariance_type='full')

gm_1.fit(X_train_1)

    

Z_1 = np.exp(gm_1.score_samples(x0.reshape(-1,1)))



plt.plot(xls0, Z_1)