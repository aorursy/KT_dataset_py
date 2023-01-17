import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import random

import datetime



from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)

train.head(10)
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
plt.figure(figsize=(25,10))

X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(10):

    rand = random.randrange(len(X_train))

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[rand], cmap=plt.get_cmap('gray'))

    plt.title(y_train[rand]);

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
classifier = KNeighborsClassifier(n_neighbors=11)

classifier.fit(X_train, y_train)
rand = random.randrange(len(X_train))

plt.imshow(X_train[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))

plt.title("real: %s , predicted : %s" % (y_train[rand], classifier.predict(np.reshape(X_train[rand], (1, -1)))));
plt.figure(figsize=(18,15))

for i in range(20):

    plt.subplot(4, 5, i + 1)

    rand = random.randrange(len(X_train))

    plt.imshow(X_train[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.title("predicted : %s" % (classifier.predict(np.reshape(X_train[rand], (1, -1)))));
now = datetime.datetime.now()

labels = classifier.predict(X_test[:100]) # probably its gonna take too much time

print((datetime.datetime.now() - now).microseconds)

labels