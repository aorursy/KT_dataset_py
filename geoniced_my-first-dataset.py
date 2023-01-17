import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots

from sklearn.neighbors import KNeighborsClassifier # Closest neighbors 

from sklearn.metrics import accuracy_score # Accuracy metrics

from sklearn.neural_network import MLPClassifier



import os

print(os.listdir("../input"))
# Loading data

mnist_train = pd.read_csv("../input/mnist_train.csv")

mnist_test  = pd.read_csv("../input/mnist_test.csv")



# Initialising column headers

cols = ["label"]

for i in range(784):

    cols.append("px_{}".format(i + 1))



# Labeling the columns

mnist_train.columns = cols

mnist_test.columns  = cols
# Get image from row of numbers

image_row_id  = 199

image_row = mnist_train.values[image_row_id, 1:] # image_row.shape = (784,)

# Reshape the row into 28x28 matrix

image_shaped = image_row.reshape(28, 28)

# Show image

plt.imshow(image_shaped, cmap="Greys")
# Getting train and test data values

# train - data from which the model will train,             we will train from it

# test  - data from which the model will test its accuracy, we will check if true values and predicted values are correct from it

train_data = mnist_train.values[:, 1:] # (60000, 784) 60000x784 matrix

test_data  = mnist_test.values[:, 1:]  # (10000, 784) 10000x784 matrix



# Getting labels for train and test data values

# Labels are the real values of data, the representation of what the data holds in itself

# Labels here are the digits from 0 to 9, and the data is 28x28 grid with grayscale numbers from 0 to 255

train_label = mnist_train.values[:, 0] # (60000,) 60000x1 matrix / vector column

test_label  = mnist_test.values[:, 0]  # (10000,) 10000x1 matrix / vector column
print(train_data.shape, train_label.shape)

print(test_data.shape, test_label.shape)
# Closest neighbors algorithm usage

# n_jobs = how many cpu cores will be used

#     -1 = all you have

# In these steps we train the model using TRAIN DATA (train_data)

# by giving it our dataset with values (28x28) and giving it correct values as labels so they map like this:

# label => 28x28 pic

# 7     => 1x784 matrix / vector row of pixels that has shape of a 7

# 3     => 1x784 shape of 3

# 2     => 1x784 shape of 2

# etc

# And then we will test what our trained model will predict if we give her

# some test data she hadn't seen before

# So the model will take her knowledge and tries to PREDICT what it will be

kn_classifier = KNeighborsClassifier(n_jobs=-1)

# .fit(x, y): fit the model using X as train data and Y as target values

kn_classifier = kn_classifier.fit(train_data, train_label)
# We should look up onto our test data and check by ourselves 

# what the random-picked row (digit) contains

test_row_id = 199

test_row_matrix = test_data[test_row_id, :].reshape(28, 28) # We taking the test_row_id ROW, and ALL the COLUMNS the test_data row contains

                                                            # and reshape it so it will like like 28x28 matrix of colormapped color brightness

plt.imshow(test_row_matrix, cmap="Greys")

print('The digit on a plot is: {}'.format(test_label[test_row_id]))
# Then, we are trying to predict with Closest neighbors classifier on a same number 

# predict(labels) - predict the class LABELS for provided data

kn_classifier.predict(test_data[test_row_id, :].reshape(1, 784))
# So on, using metrics we doing the same stuff with all test data set

kn_predictions = kn_classifier.predict(test_data)

# And output of a total score

total_score = accuracy_score(test_label, kn_predictions)

print("Точность модели: {}".format(total_score * 100))
# Neural network

# Same as with closest neighbors, we fit, we predict

# verbose = Do output

# max_iter = max iterations

# n_iter_no_change = how many iterations without breaking the delta there should be 

mlp_classifier = MLPClassifier(verbose=True, max_iter=750, n_iter_no_change=700)

mlp_classifier = mlp_classifier.fit(train_data, train_label)
# Now we predict the number using neural network Multi-layer perceptron

mlp_classifier.predict(test_data[test_row_id, :].reshape(1, 784))
# Now we predict all the numbers!

# And check the accuracy score!

mlp_predictions = mlp_classifier.predict(test_data)

mlp_total_score = accuracy_score(test_label, mlp_predictions)

print("Точность модели MLP: {}".format(mlp_total_score * 100))