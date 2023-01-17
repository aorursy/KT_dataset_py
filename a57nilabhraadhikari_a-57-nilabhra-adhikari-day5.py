#Write a Python program using Scikit-learn to print the keys, number of rows-columns, feature names and the description of the Iris data.

import pandas as pd

iris_data = pd.read_csv("../input/iriscsv/Iris.csv")

print("\nKeys of Iris dataset:")

print(iris_data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(iris_data.shape) 
# Write a Python program to get the number of observations, missing values and nan values.

import pandas as pd

iris = pd.read_csv("../input/iriscsv/Iris.csv")

print(iris.info())
#Write a Python program to create a 2-D array with ones on the diagonal and zeros elsewhere.

import numpy as np

from scipy import sparse

eye = np.eye(4)

print("NumPy array:\n", eye)

sparse_matrix = sparse.csr_matrix(eye)

print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
#Write a Python program to load the iris data from a given csv file into a dataframe and print the shape of the data, type of the data and first 3 rows.

import pandas as pd

data = pd.read_csv("../input/iriscsv/Iris.csv")

print("Shape of the data:")

print(data.shape)

print("\nData Type:")

print(type(data))

print("\nFirst 3 rows:")

print(data.head(3))