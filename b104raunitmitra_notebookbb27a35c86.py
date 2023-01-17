
import pandas as pd
data = pd.read_csv("../input/iriscsv/Iris.csv")
print("\nThe Keys of Iris data:")
print(data.keys())
print("\n\nShape of the data:")
print(data.shape)
print(data.columns)
print("\nData Type:")
print(type(data))
print (data.info())
from scipy import sparse
import numpy as np
eye = np.eye(5)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)

print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))