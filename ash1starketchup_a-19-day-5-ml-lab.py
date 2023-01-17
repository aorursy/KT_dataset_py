import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")
print("\nHere are the 'keys' of the Iris dataset:-")
print(data.keys())
print("\nHere are the number of Rows and Columns in the Iris dataset:")
print(data.shape) 
# Here's the datatype of the Iris dataset 
print(type(data))
# Here's a statistical description of the Iris Dataset
data.describe()
print(data.info())
# information provided in the dataset
data.count()
# no. of rows per series (i.e. columns) = 150 (HERE)
data.isnull() # Shows the null values
# Counts the null values per column (series)
data.isnull().sum()
data.isnull().values.any()
# 'False' indicates NAN not found i.e. no NAN values
import numpy as np
from scipy import sparse
size = int(input("Enter the size of the matrix here:-"))
eye = np.eye(size)
print("NumPy array:\n", eye)
import pandas as pd
df= pd.read_csv("../input/iris/Iris.csv")
print("Shape of the dataset:")
print(df.shape)
print("\nHere's the datatype:")
print(type(df))
print("\nHere are the first 3 rows of the Iris dataset:")
print(df.head(3))
