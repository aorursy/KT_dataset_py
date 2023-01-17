#1

import pandas as pd 

iris_data = pd.read_csv("../input/iris-dataset/iris.data.csv")

print("\nKeys of Iris dataset:")

print(iris_data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(iris_data.shape)
#2

import pandas as pd

iris = pd.read_csv("../input/iris-dataset/iris.data.csv")

print(iris.info())

#3

import numpy as np

from scipy import sparse

eye = np.eye(4)

print("NumPy array:\n", eye)

sparse_matrix = sparse.csr_matrix(eye)

print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
#4

import pandas as pd

data = pd.read_csv("../input/iris-dataset/iris.data.csv")

print("Shape of the data:")

print(data.shape)

print("\nData Type:")

print(type(data))

print("\nFirst 3 rows:")

print(data.head(3))