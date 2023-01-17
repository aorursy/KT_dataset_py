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
#A pulsar (from pulse and -ar as in quasar) is a highly magnetized rotating neutron star that emits beams of electromagnetic radiation out of its magnetic poles. 

#This radiation can be observed only when a beam of emission is pointing toward Earth (much like the way a lighthouse can be seen only when the light is pointed in the direction of an observer).

#It is responsible for the pulsed appearance of emission. Neutron stars are very dense, and have short, regular rotational periods. 

#This produces a very precise interval between pulses that ranges from milliseconds to seconds for an individual pulsar. Pulsars are one of the candidates for the source of ultra-high-energy cosmic rays (see also centrifugal mechanism of acceleration).
#impoting the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# importing the dataset values

dataset = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')    #uses Standard deviation of integrated profile & Excess Kurtosos of integrated profile as inputs

X= dataset.iloc[:, 1:3].values  

#choosing the fields with which we want to predict the pulsar,here i choose only first 2 fields .

y = dataset.iloc[:, -1].values



#class of pulsar star----------------------------------------VERY IMP: 1 for pulsar star,0 for not a star



# Splitiing the evalues between training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Scaling the values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#USING K_NEAREST_NEIGHBOUR CLASSIFIER ALGORITHM

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



#visualizing our training data 

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('K-NN (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('K-NN (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()
# USING KERNEL_SVM ALGORITHM

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



#visualizing training set

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('KERNEL_SVM (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('KERNEL_SVM (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()
#USING SUPPORT VECTOR MACHINE ALGORITHM

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



#visualizing the training set

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('SVC (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('SVC (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()
#USING THE NAIVE_BAYES ALGORITHM

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



# visualizing our training set

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('NAIVE_BAYES (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('NAIVE_BAYES (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()
# USING DECISION TREE ALGORITHM

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



# visualizing our training set

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('DECISION TREE (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('DECISION TREE (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()
#USING RANDOM FOREST CLASSIFIER ALGORITHM

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))

print(cm)

print(accuracy_score(y_test, classifier.predict(X_test)))



# visualizing our training set

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),  #increment range of 5units on both sides

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.2))      #increment range of 1 unit on both sides

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.5, cmap = ListedColormap(('cyan', 'magenta')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'orange'))(i), label = j)

plt.title('RANDOM FOREST (Training set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()



# visulaizing our test data

from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max()+ 1, step = 0.5))

plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('RANDOM FOREST (Test set)')

plt.xlabel('Standard deviation of integrated profile')

plt.ylabel('Excess Kurtosos of integrated profile')

plt.legend()

plt.show()