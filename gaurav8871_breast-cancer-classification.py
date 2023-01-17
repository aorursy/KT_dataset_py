# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# ============================== loading libraries ===========================================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# =============================================================================================
# ============================== data preprocessing ===========================================

# loading the dataset

data = pd.read_csv('../input/data.csv')
# rows * columns

data.shape
# column labels

data.columns
# data of top 5 rows

data.head()
# unique values of the column which is to be predicted by the classifier

data['diagnosis'].unique()
# total number of null values in every column

data.isna().sum()
# dropping the null values

data.dropna(axis=1, inplace=True)
# Malignant = 0 

# Benign = 1

data['diagnosis'] = data.diagnosis.map(lambda x: 0 if x == 'M' else 1)
data['diagnosis'].unique()
# convert the dataset into numpy's ndarray (X and y)

y = data['diagnosis'].values

data.drop(['diagnosis', 'id'], inplace=True, axis=1)

X = data.values

print(type(X))

print(type(y))
# split the data set into train and test

X1, X_test, y1, y_test = train_test_split(X, y, test_size=0.2)
# split the train data set into cross validation train and cross validation test

X_train, X_cv, y_train, y_cv = train_test_split(X1, y1, test_size=0.2)

# =============================================================================================
# ====================== Finding the optimal value of K for K-NN ===============================

# list to store accuracy_score

accuracy = []

# dict to store accuracy as key and k-value as value

accuracy_dict = {}

for i in range(1,30,2):

    # instantiate the K-NN classifier with k = i

    clf = KNeighborsClassifier(n_neighbors=i)

    # fitting the model with training data

    clf.fit(X_train, y_train)

    # append the accuracy_score of cross_validation data into accuracy list

    acc = accuracy_score(y_cv, clf.predict(X_cv)) * float(100)

    accuracy.append(acc)

    accuracy_dict[acc] = i



# plot the accuracy and the value of K to findout optimal value for K

plt.xlabel('Value of K')

plt.ylabel('Accuracy Score')

k = [i for i in range(1,30,2)]

plt.plot(k,accuracy)

# =============================================================================================
# ====================== Fitting the model with optimal value of K ============================

# getting the highest accuracy

acc = sorted(accuracy)[-1]

# getting the k-value of the highest accuracy

k_val = accuracy_dict[acc]

# instantiating the K-NN model

clf = KNeighborsClassifier(n_neighbors=k_val)

# fitting the model

clf.fit(X_train, y_train)

# checking the accuracy by predicting the y_test

print(accuracy_score(y_test, clf.predict(X_test)) * float(100))

# =============================================================================================