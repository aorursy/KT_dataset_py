import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

import pandas as pd

import random

import itertools

import seaborn as sns

from numpy.linalg import inv

import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

matplotlib.style.use('fivethirtyeight')

from scipy import ndimage as ndi

import matplotlib.pylab as plt





data = pd.read_csv('../input/data.csv')

labels = data.diagnosis

labelid = data.id

A = data.iloc[:,2:31]





#plut U-factors

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(A, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()

A=A.transpose()

# singular value decomposition

U,s,V=np.linalg.svd(A,full_matrices=False)







#plut U-factors

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(U, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()
#plot V factors

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(V, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()
S=np.diag(s)

iS=inv(S)



#plot S

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(S, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()
US=np.dot(U,iS)

#plot US

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(US, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()