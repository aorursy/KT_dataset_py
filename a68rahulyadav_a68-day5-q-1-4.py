#Q1.printing the keys, number of rows-columns, feature names 

#and the description of the Iris data.

import pandas as pd

iris_data = pd.read_csv("../input/iris-dataset/iris.data.csv")

print("\nKeys of Iris dataset:")

print(iris_data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(iris_data.shape)

print("Data type:")

print(type(iris_data))
#Q2.get the number of observations, missing values and nan values.

print("No.of Observations are:")

print(iris_data.count().sum())

print("No. of Nan is:")

print(iris_data.isnull().sum())
#Q3.create a 2-D array with ones on the diagonal and zeros elsewhere.

import numpy as np

from scipy import sparse

eye = np.eye(5)

print("NumPy array:\n", eye)

sparse_matrix = sparse.csr_matrix(eye)

print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
#Q4.load the iris data from a given csv file into a dataframe and 

#print the shape of the data, type of the data and first 3 rows.

print("Shape of the data:")

print(iris_data.shape)

print("\nData Type:")

print(type(iris_data))

print("\nFirst 3 rows:")

print(iris_data.head(3))