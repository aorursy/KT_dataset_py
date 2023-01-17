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
#Program 1

import pandas as pd

data = pd.read_csv("../input/iris-dataset/iris.data.csv")

print("\nKeys of Iris dataset:")

print(data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(data.shape) 
#Program 2

data.info()
#Program 3

import numpy as np

Matrix = np.eye(4)

print("NumPy array:\n", Matrix)
#Program 4

import pandas as pd

data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

print("Shape of the data:")

print(data.shape)

print("\nData Type:")

print(type(data))

print("\nFirst 3 rows:")

print(data.head(3))
