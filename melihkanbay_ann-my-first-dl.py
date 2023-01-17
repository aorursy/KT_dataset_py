# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
x_l = np.load("/kaggle/input/sign-language-digits-dataset/X.npy")

y_l = np.load("/kaggle/input/sign-language-digits-dataset/Y.npy")

img_size = 64

plt.subplot(1,2,1)

plt.imshow(x_l[260].reshape(img_size,img_size))

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(x_l[900].reshape(img_size,img_size))

plt.axis("off")
X = np.concatenate((x_l[204:409], x_l[822:1027]),axis = 0)

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0],1)

X.shape,Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.15, random_state = 42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]
x_train= X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

x_test = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

x_train.shape,x_test.shape
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 100)

print("test accuracy: {} ".format(logreg.fit(x_train, Y_train).score(x_test, Y_test)))
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense
def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation = "relu", input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation = "relu"))

    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = Y_train, cv = 3)

accs = accuracies

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracys: "+str(accs))

print("Accuracys mean: "+str(mean))

print("Variance: "+str(variance))