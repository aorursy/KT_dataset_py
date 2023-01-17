import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
#load data

df = pd.read_csv("../input/train.csv")
# Take out label column

labels = df['label']

df = df.drop('label', 1)
# Let's see how data looks like 

image = df.values[123]

plt.imshow(image.reshape(28, 28), cmap='Greys')

plt.show()
# Split data

x_train, x_test, y_train, y_test = train_test_split(df.values, labels, test_size = 0.3, random_state = 0)



print(x_train.shape)

print(x_test.shape)



print(y_train.shape)

print(y_test.shape)
# Setup SVM model

clf = svm.SVC(gamma=0.001, kernel='poly')
#Train model

clf.fit(x_train, y_train)
# Predict

predictions = clf.predict(x_test)
# evaluate predictions

print ("Overal accuracy:", accuracy_score(predictions, y_test))

print (classification_report(predictions, y_test))

print (confusion_matrix(predictions, y_test))