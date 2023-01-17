import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



from sklearn.naive_bayes import BernoulliNB
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



target = train['label']

train.drop('label', axis=1)



train.head()



plt.imshow(np.resize(train[0]), (28, 28))
nb = BernoulliNB()

y_pred = nb.fit(train, target).predict(train)

accuracy = (target == y_pred).sum() / 42000 * 100

print("Accuracy: %d" % accuracy)

print("Number of mislabeled points out of a total %d points : %d" % (train.shape[0], (target != y_pred).sum()))
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD
model = Sequential()