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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
# (a) - Load data and report sizes of training and testing datasets

train_data = pd.read_csv('../input/pcadata/pca_train.csv')

test_data = pd.read_csv('../input/pcadata/pca_test.csv')



print('Size of Training dataset: ' + str(train_data.shape))

print('Size of Testing dataset: ' + str(test_data.shape))
# To identify the number of Class 0 and Class 1 samples in the training and testing sets

train_class_0 = train_data.loc[train_data['Class'] == 0].shape[0]

train_class_1 = train_data.loc[train_data['Class'] == 1].shape[0]

test_class_0 = test_data.loc[test_data['Class'] == 0].shape[0]

test_class_1 = test_data.loc[test_data['Class'] == 1].shape[0]



print('\nTraining dataset:')

print('Number of samples with Class (0): ' + str(train_class_0))

print('Number of samples with Class (1): ' + str(train_class_1))



print('\nTesting dataset:')

print('Number of samples with Class (0): ' + str(test_class_0))

print('Number of samples with Class (1): ' + str(test_class_1))



# Normalize training data

for col in test_data.columns:

  if col != 'Class':

    test_data[col] = (test_data[col] - min(train_data[col]))/(max(train_data[col]) - min(train_data[col]))



# Normalize testing data

for col in train_data.columns:

  if col != 'Class':

    train_data[col] = (train_data[col] - min(train_data[col]))/(max(train_data[col]) - min(train_data[col]))
train_covariance_matrix = train_data.iloc[:, :-1].cov()

print('\nDimensions of co-variance matrix of training data set: ' + str(train_covariance_matrix.shape))

print('\nPrinting first 5 rows and first 5 columns of the covariance matrix: ')

print(str(train_covariance_matrix.iloc[:5, :5]))


w, v = np.linalg.eig(train_covariance_matrix)

print('Size of co-variance matrix: ' + str(train_covariance_matrix.size))

print('Size of eigen-vectors: ' + str(v.size))

print('5 largest eigen values: ' + str(w[0:5]))
plt.bar(np.arange(w.shape[0]), w)

plt.xlabel("Component Number") 

plt.ylabel("Eigen Value") 

plt.title("Bar plot of Eigen values") 

plt.show()
p = [2, 4, 8, 10, 20, 40, 60]



# Using sklearn's KNeighborsClassifier for 5 neighbors and Euclidean distance metric

knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

test_accuracy = {}

normal_train_x = train_data.iloc[:, :-1]

normal_test_x = test_data.iloc[:, :-1]



for p_comp_num in p:

  # reducing dimensionality of training dataset

  transformed_train_data_attributes = normal_train_x.dot(v[:, :p_comp_num])

  # fitting the 5NN model

  model = knn_classifier.fit(transformed_train_data_attributes, train_data['Class'])

  # creating new test dataset using PCA

  transformed_test_data_attributes = normal_test_x.dot(v[:, :p_comp_num])

  # using the 5NN model to predict values

  predicted_values = model.predict(transformed_test_data_attributes)

  # calculating test accuracy

  result = metrics.accuracy_score(test_data.iloc[:, -1], predicted_values)

  test_accuracy[p_comp_num] = result



  # (b) - (iv) - Point-1: Creating new_testing_dataset.csv for p=4

  if p_comp_num == 4:

    # (b) - (iv) - Point-1: Reposrting testing accuracy for p=4

    print('Accuracy of NEW testing dataset when using PCA(p=4) with 5NN: ' + str(test_accuracy[4]))

    test_p_4_df = transformed_test_data_attributes

    test_p_4_df['true_Class'] = test_data.iloc[:, -1]

    test_p_4_df['predicted_Class'] = predicted_values

    test_p_4_df.to_csv('new_testing_dataset.csv')
x = []

y = []

for k, v in test_accuracy.items():

  x.append(k)

  y.append(v)



plt.plot(x, y, marker='o')

plt.xlabel('Number of components (p)')

plt.ylabel('Test Accuracy')

plt.title('Test Accuracy Plot')

plt.show()
std_train_data = pd.read_csv('../input/pcadata/pca_train.csv')

std_test_data = pd.read_csv('../input/pcadata/pca_test.csv')
for col in std_test_data.columns:

  if col != 'Class':

    std_test_data[col] = (std_test_data[col] - std_train_data.mean(axis = 0)[col])/std_train_data[col].std()



for col in std_train_data.columns:

  if col != 'Class':

    std_train_data[col] = (std_train_data[col] - std_train_data.mean(axis = 0)[col])/std_train_data[col].std()
train_covariance_matrix = std_train_data.iloc[:, :-1].cov()

print('\nDimensions of co-variance matrix of training data set: ' + str(train_covariance_matrix.shape))

print('\nPrinting first 5 rows and first 5 columns of entire covariance matrix: ')

print(str(train_covariance_matrix.iloc[:5, :5]))


w, v = np.linalg.eig(train_covariance_matrix)

print('Size of co-variance matrix: ' + str(train_covariance_matrix.size))

print('Size of eigen-vectors: ' + str(v.size))

print('5 largest eigen values: ' + str(w[0:5]))




plt.bar(np.arange(w.shape[0]), w)

plt.xlabel("Component Number") 

plt.ylabel("Eigen Value") 

plt.title("Bar plot of Eigen values") 

plt.show()

p = [2, 4, 8, 10, 20, 40, 60]



# Using sklearn's KNeighborsClassifier for 5 neighbors and Euclidean distance metric

knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

test_accuracy = {}

normal_train_x = std_train_data.iloc[:, :-1]

normal_test_x = std_test_data.iloc[:, :-1]



for p_comp_num in p:

  # reducing dimensionality of training dataset

  transformed_train_data_attributes = normal_train_x.dot(v[:, :p_comp_num])

  # fitting the 5NN model

  model = knn_classifier.fit(transformed_train_data_attributes, train_data['Class'])

  # creating new test dataset using PCA

  transformed_test_data_attributes = normal_test_x.dot(v[:, :p_comp_num])

  # using the 5NN model to predict values

  predicted_values = model.predict(transformed_test_data_attributes)

  # calculating test accuracy

  result = metrics.accuracy_score(std_test_data.iloc[:, -1], predicted_values)

  test_accuracy[p_comp_num] = result



  # (c) - (iv) - Point-1: Creating new_testing_dataset.csv for p=4

  if p_comp_num == 4:

    # (c) - (iv) - Point-1: Reposrting testing accuracy for p=4

    print('Accuracy of NEW testing dataset when using PCA(p=4) with 5NN: ' + str(test_accuracy[4]))

    test_p_4_df = transformed_test_data_attributes

    test_p_4_df['true_Class'] = std_test_data.iloc[:, -1]

    test_p_4_df['predicted_Class'] = predicted_values

    test_p_4_df.to_csv('new_standardize_testing_dataset.csv')



    print(test_accuracy)
x = []

y = []

for k, v in test_accuracy.items():

  x.append(k)

  y.append(v)



plt.plot(x, y, marker='o')

plt.xlabel('Number of components (p)')

plt.ylabel('Test Accuracy')

plt.title('Test Accuracy Plot')

plt.show()