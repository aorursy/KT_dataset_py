# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import modules

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA 

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from random import randint



# import train and test datasets

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# print dataset shapes

print('train set shape = ', train.shape)

print('test set shape = ', test.shape)

# graph a sample

plt.figure(), plt.imshow(np.reshape(train.iloc[randint(0,train.shape[0] - 1),1:].values, (28,28)), cmap = 'Greys')



# create feature matrices

Xtrain = train.iloc[:,1:]

Xtest = test.iloc[:,:]

# create target array

ytrain = train.iloc[:,0]

# print matrices shapes

print('train features shape = ', Xtrain.shape, '\ntest features shape = ', Xtest.shape, '\ntrain target shape = ', ytrain.shape)



# create PCA scaler using variance threshold

pca_scaler = PCA(0.98)

# dimensionality reduction

Xtrain = pca_scaler.fit_transform(Xtrain)

Xtest = pca_scaler.transform(Xtest)

# number of components used

print('Dimensionality was reduce using ', pca_scaler.n_components_, ' PCA components.')



# create estimator

clf = RandomForestClassifier(n_estimators = 100, max_depth = 20)

# fit model

clf.fit(Xtrain, ytrain)

# predictions

predictions = clf.predict(Xtest)

# export submission

pd.DataFrame({'ImageId':np.squeeze(np.array([range(1,28001)])), 'Label': predictions}).to_csv('my_submission.csv', index = False)