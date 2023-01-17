import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")
print("\nKeys of Iris dataset:")
print(data.keys())
print("\nNumber of rows and columns of Iris dataset:")
print(data.shape) 
import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")
print(data.info())
import numpy as np
arr = np.eye(4)
print("Array:\n", arr)
import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")
print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))
