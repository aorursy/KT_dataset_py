#printing the keys, number of rows-columns, feature names and the description of the Iris data.
import pandas as pd
data = pd.read_csv("../input/iriscsv/Iris.csv")
print("\nThe Keys of Iris data:")
print(data.keys())
print("\n\nShape of the data:")
print(data.shape)
print(data.columns)
print("\nData Type:")
print(type(data))
#get the number of observations, missing values and nan values.
print(data.info())
#create a 2-D array with ones on the diagonal and zeros elsewhere.
from scipy import sparse
import numpy as np
eye = np.eye(5)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
#load the iris data from a given csv file into a dataframe and print the shape of the data, type of the data and first 3 rows.
print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))
