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

X = array[:,0:2]

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
X_train_0_mean = [X_train_0[:,0:1].mean(), X_train_0[:,1:2].mean()]

print('X_train mean of class 0 ' + str(X_train_0_mean))

#X_train_0_var = [X_train_0[:,0:1].var(), X_train_0[:,1:2].var()]

#print('X_train var of class 0 ' + str(X_train_0_var))



X_train_1_mean = [X_train_1[:,0:1].mean(), X_train_1[:,1:2].mean()]

print('X_train mean of class 1 ' + str(X_train_1_mean))

#X_train_1_var = [X_train_1[:,0:1].var(), X_train_1[:,1:2].var()]

#print('X_train var of class 1 ' + str(X_train_1_var))



df_0 = pandas.DataFrame(X_train_0)



df_1 = pandas.DataFrame(X_train_1)
X_train_min = [X_train[:,0:1].min(), X_train[:,1:2].min()]

X_train_max = [X_train[:,0:1].max(), X_train[:,1:2].max()]



print('X_train min val ' + str(X_train_min))

print('X_train max val ' + str(X_train_max))
xls0 = np.linspace(X_train_min[0], X_train_max[0])

xls1 = np.linspace(X_train_min[1], X_train_max[1])



#x0 = X_train_0[:,0:1]

#x1 = X_train_0[:,1:2]



x0, x1 = np.meshgrid(xls0, xls1)



pos = np.empty(x0.shape + (2,))

pos[:, :, 0] = x0

pos[:, :, 1] = x1



mn_0 = multivariate_normal(mean=X_train_0_mean, cov=df_0.cov())

pdf_0 = mn_0.pdf(pos)



mn_1 = multivariate_normal(mean=X_train_1_mean, cov=df_1.cov())

pdf_1 = mn_1.pdf(pos)
# Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, pdf_0, rstride=3, cstride=3, linewidth=1, antialiased=True,

                cmap=cm.viridis)



cset = ax.contourf(x0, x1, pdf_0, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.6)

ax.set_zticks(np.linspace(0,0.6,6))

ax.view_init(27, -21)



plt.show()
# Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, pdf_1, rstride=3, cstride=3, linewidth=1, antialiased=True,

                cmap=cm.viridis)



cset = ax.contourf(x0, x1, pdf_1, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.6)

ax.set_zticks(np.linspace(0,0.6,6))

ax.view_init(27, -21)



plt.show()
kde_0 = KernelDensity(bandwidth=0.2, kernel='epanechnikov')

kde_0.fit(X_train_0, 0)



Z_0 = []

for i in range(0, xls0.size):

    x_sample = []

    for j in range(0, xls0.size):

        x_sample.append([x0[i,j], x1[i,j]])

    x_sample = np.array(x_sample)

    t = np.exp(kde_0.score_samples(x_sample))

    Z_0.append(t)

    

Z_0 = np.array(Z_0)

    

# 1



kde_1 = KernelDensity(bandwidth=0.2, kernel='epanechnikov')

kde_1.fit(X_train_1, 1)



Z_1 = []

for i in range(0, xls0.size):

    x_sample = []

    for j in range(0, xls0.size):

        x_sample.append([x0[i,j], x1[i,j]])

    x_sample = np.array(x_sample)

    t = np.exp(kde_1.score_samples(x_sample))

    Z_1.append(t)

    

Z_1 = np.array(Z_1)
# Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, Z_0, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)



cset = ax.contourf(x0, x1, Z_0, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.9)

ax.set_zticks(np.linspace(0,0.9,9))

ax.view_init(27, -21)



plt.show()
# Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, Z_1, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)



cset = ax.contourf(x0, x1, Z_1, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.9)

ax.set_zticks(np.linspace(0,0.9,9))

ax.view_init(27, -21)



plt.show()
gm_0 = mixture.GaussianMixture(n_components=3, covariance_type='full')

gm_0.fit(X_train_0)



Z_0 = []

for i in range(0, xls0.size):

    x_sample = []

    for j in range(0, xls0.size):

        x_sample.append([x0[i,j], x1[i,j]])

    x_sample = np.array(x_sample)

    t = np.exp(gm_0.score_samples(x_sample))

    Z_0.append(t)

    

Z_0 = np.array(Z_0)

    

gm_1 = mixture.GaussianMixture(n_components=3, covariance_type='full')

gm_1.fit(X_train_1)



Z_1 = []

for i in range(0, xls0.size):

    x_sample = []

    for j in range(0, xls0.size):

        x_sample.append([x0[i,j], x1[i,j]])

    x_sample = np.array(x_sample)

    t = np.exp(gm_1.score_samples(x_sample))

    Z_1.append(t)

    

Z_1 = np.array(Z_1)
# Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, Z_0, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)



cset = ax.contourf(x0, x1, Z_0, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.9)

ax.set_zticks(np.linspace(0,0.9,9))

ax.view_init(27, -21)



plt.show()
#Create a surface plot and projected filled contour plot under it.

fig = plt.figure()

ax = fig.gca(projection='3d', xlabel = 'x0', ylabel = 'x1')

ax.plot_surface(x0, x1, Z_1, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)



cset = ax.contourf(x0, x1, Z_1, zdir='z', offset=0, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(0,0.9)

ax.set_zticks(np.linspace(0,0.9,9))

ax.view_init(27, -21)



plt.show()