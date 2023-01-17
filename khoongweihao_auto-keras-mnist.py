!pip install autokeras
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline



from keras.datasets import mnist

from autokeras.image.image_supervised import ImageClassifier
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0

test = test/255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# plotting the first five training images

fig = plt.figure(figsize=(20,20))

for i in range(5):

    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(28,28), cmap='gray')

    ax.set_title(str(Y_train[i]))
# print the first five encoded training labels

print('One-hot Encoded labels:')

print(Y_train[:10])
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print(X_train.shape, Y_train.shape)
clf = ImageClassifier(verbose=True)

clf.fit(X_train, Y_train, time_limit=3 * 60 * 60)

clf.final_fit(X_train, Y_train, X_val, Y_val, retrain=True)

Y = clf.evaluate(X_val, Y_val)

print(Y)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = Y_val

submission.to_csv('submission.csv',index=False)