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
#question 2

import pandas as pd

df=pd.read_csv("../input/iris-dataset/iris.data.csv")

print("no.of observation is :")

print(df.count().sum())

print("no.of nan is :")

print(df.isnull().sum())
#question 1



import pandas as pd

iris_data = pd.read_csv("../input/iris-dataset/iris.data.csv")

print("\nKeys of Iris dataset:")

print(iris_data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(iris_data.shape)
#question 3



import pandas as pd

diagonal=4

array2d=np.eye(diagonal)

print(array2d)
#question 4



import pandas as pd

data=pd.read_csv("../input/iris-dataset/iris.data.csv")

print("shape of data is :")

print(data.shape)

print("\n data type  is :")

print(type(data))

print("first three rows are :")

print(data.head(3))