import pandas as pd
data=pd.read_csv('../input/iris/Iris.csv')
data.shape
data.dtypes
data.iloc[:3]
import numpy as np
from scipy import sparse
sz = int(input("Enter the size of the matrix here:-"))
arr = np.eye(sz)
print("NumPy array:", arr)
data.count()
data.isnull().sum()
data.keys()
data.shape
data.columns
data.describe()
