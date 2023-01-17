# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
#Question 1
#Write a Python program using Scikit-learn to print the keys, number of rows-columns, feature names and the description of the Iris data.

import pandas as pd
data=pd.read_csv("../input/iris/Iris.csv")
print("\nKeys of Iris dataset:")
print(data.keys())
print("\nNumber of rows and columns of Iris dataset:")
print(data.shape) 
#Question 2
#Write a Python program to get the number of observations, missing values and nan values.

data.info()
#Question 3
#Write a Python program to create a 2-D array with ones on the diagonal and zeros elsewhere.

import numpy as np
matrix=np.eye(5)
print("NumPy array:\n", matrix)
#Question 4
#Write a Python program to load the iris data from a given csv file into a dataframe and print the shape of the data, type of the data and first 3 rows.

print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))