# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import arff



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Trainingsdaten laden

dataset = arff.load(open('../input/classifier-train/train.arff'))

data = np.array(dataset['data'])

data = data.astype(np.float64)



X_train = data[:, [0, 1]]

#print(X_train)



y_train = np.array(data[:, -1])

#print(y_train)



plt.figure(figsize=(8,6))

sns.scatterplot(x=X_train[:,0], y=X_train[:,1],hue=y_train.astype(int))

plt.title("Trainingsdaten")

plt.show()
#Testdaten laden

test_dataset = arff.load(open('../input/classifier-test/eval.arff'))

test_data = np.array(test_dataset['data'])

test_data = test_data.astype(np.float64)



X_test = test_data[:, [0,1]]

y_test = np.array(data[:,-1])



neighbors = np.arange(1, len(y_test))

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)



    # Accuracy der Trainings - und Testdaten für die Anzahl an Nachbarn speichern

    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)



plt.figure(figsize=(12,6))

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()



print("Highest Accuracy:", test_accuracy[test_accuracy.argmax()], "with", test_accuracy.argmax() + 1, "neighbors")
# Daten laden

circle_data = pd.read_csv('../input/circledataset/train.csv')





X_train = circle_data.drop(['class'],axis=1)

y_train = circle_data['class']

circle_data.head()
# Daten visualisieren:

plt.figure(figsize=(8,6))

sns.scatterplot(x=circle_data['X'], y=circle_data['Y'], hue=circle_data['class'])

plt.title("Circle - Trainingsdaten")

plt.show()
test_data = pd.read_csv('../input/circledataset/test.csv')

X_test = test_data.drop(['class'],axis=1)

y_test = test_data['class']



plt.figure(figsize=(8,6))

sns.scatterplot(x='X',y='Y', data=test_data, hue='class')

plt.show()
neighbors = np.arange(1, len(y_test))

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)



    # Accuracy der Trainings - und Testdaten für die Anzahl k an Nachbarn speichern

    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)



plt.figure(figsize=(12,6))

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()



print("Highest Accuracy:", test_accuracy[test_accuracy.argmax()])