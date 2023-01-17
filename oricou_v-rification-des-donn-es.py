import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt



%matplotlib inline
!ls ../input/
types = ['containership', 'cruiser', 'destroyer','coastguard', 'smallfish', 'methanier', 'cv', 'corvette', 'submarine', 'tug']

types_id = dict(zip(types, range(len(types))))



ships = np.load('../input/ships.npz')

X_data = ships['X']

Y_data = ships['Y']
from keras.utils import np_utils



X_train = X_data

Y_train = np_utils.to_categorical(Y_data)
ships = np.load('../input/ships_test.npz')

X_test = ships['X']

Y_test = np_utils.to_categorical(ships['Y'])
for i in range(len(X_test)):

    if np.log10(i+1) % 1 < 1E-6:

        print("i=",i)

    for j in range(len(X_train)):

        if (X_test[i] == X_train[j]).all():

            print(i,j)
i = 6400

print("Ship #%d is a %s" % (i,types[np.argmax(Y_train[i])]))

print(Y_train[i])

plt.imshow(X_train[i])
i = 0

print("Ship #%d is a %s" % (i,types[np.argmax(Y_test[i])]))

print(Y_test[i])

plt.imshow(X_test[i])