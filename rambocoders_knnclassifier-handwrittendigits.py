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
# import the necessary packages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.datasets import mnist
#from skimage import exposure
import pandas as pd
import numpy as np
# import imutils
import matplotlib.pyplot as plt

# load the MNIST digits dataset

(X_train,y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

#Reshape the data to fit the model
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing



# show the sizes of each data split

print("training data points: {}".format(len(y_train)))
print("validation data points: {}".format(len(y_test)))
print("testing data points: {}".format(len(y_test)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
# train the k-Nearest Neighbor classifier with the current value of `k`
model = KNeighborsClassifier(n_neighbors=200)
model.fit(X_train, y_train)
# evaluate the model and update the accuracies list

score = model.score(X_test, y_test)
print("k=%d, accuracy=%.2f%%" % (245, score * 100))
accuracies.append(score)

for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    # evaluate the model and update the accuracies list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

df_submission = pd.DataFrame([range(1, X_test.size),predictions],["ImageId","Label"]).transpose()
df_submission.to_csv("submission.csv",index=False)
# print(predictions[1])

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits

print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))

print("Confusion matrix")
print(confusion_matrix(y_test, predictions))

# loop over a few random digits

