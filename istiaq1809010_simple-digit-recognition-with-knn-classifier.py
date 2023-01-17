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
# Import necessary modules

from sklearn import datasets

import matplotlib.pyplot as plt



# Load the digits dataset: digits

digits = datasets.load_digits()



# Print the keys and DESCR of the dataset

print(digits.keys())

print(digits['DESCR'])



# Print the shape of the images and data keys

print(digits.images.shape)

print(digits.data.shape)



# Display digit 1010

plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
from sklearn.model_selection import train_test_split



# Create feature and target arrays

X = digits.data

y = digits.target



# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier

# Create a k-NN classifier with 7 neighbors: knn

knn =KNeighborsClassifier(n_neighbors=7) 

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(y_pred)
# Print the accuracy

print(str(knn.score(X_test, y_test)*100)+' %')
# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train,y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test,y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()