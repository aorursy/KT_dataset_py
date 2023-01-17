!pip install idx2numpy

!pip install sklearn

!pip install numpy

!pip install pandas



from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets

import idx2numpy

import numpy as np

import pandas as pd
# Since the data from the database was in binary format, we have to convert into numpy array.

# So, we use 'idx2numpy' module which has 'convert_from_file' function that convert the idx data to numpy array.

X_train = idx2numpy.convert_from_file('../input/train-images.idx3-ubyte')

y_train = idx2numpy.convert_from_file('../input/train-labels.idx1-ubyte')

X_test = idx2numpy.convert_from_file('../input/t10k-images.idx3-ubyte')

y_test = idx2numpy.convert_from_file('../input/t10k-labels.idx1-ubyte')



print(f'Shape of the training array : {X_train.shape} ')

print(f'Shape of the test array : {X_test.shape} ')



#Reshaping the training feature np_array and testing feature np_array since KNN classifier takes 2-d array as arguments.

#But the data is in 3-d array.

X_train = X_train.flatten()

X_train = X_train.reshape(60000,784)

X_test = X_test.flatten()

X_test = X_test.reshape(10000,784)


# Initializing KNN classifier with 7 neighbours.

# We can change n_neighbors to avoid underfitting or overfitting of data. 

# If we decrease there are chances of overfitiing.

# If we increase there are chances of underfitiing.

# Since the KNN classifier take a lot of time so to lower the time consumption we use all the cores of cpu by providing

# n_jobs=-1 as argument. If your system has lower specification change it to 1.

knn = KNeighborsClassifier(n_neighbors=7,n_jobs=-1)



# Fitting or traning the model.

knn.fit(X_train,y_train)
# Gives the Accuracy score of the model.

acc_testData= knn.score(X_test,y_test) * 100

# acc_trainData = knn.score(X_train,y_train) * 100 // Uncomment to find the accuracy of training data, But it takes a lot of time
print(f'Accuracy of Test data : {acc_testData}')

# print(f'Accuracy of Test data : {acc_trainData}') // Uncomment to print the accuracy of training data.
# Predict the digit using the unknown data(TEST DATA) provided to the model.

pred = knn.predict(X_test[0].reshape(1, -1))

print(f'Actual digit in test data : {y_test[0]}')

print(f'Predicted digit by the classifier : {pred}')