# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import nan

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import glob
import os
import datetime
import time
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# See how many files there are in the directory. 
# "!" commands are called "magic commands" and allow you to use bash
file_dir = '../input/uvtownsville'
!ls $file_dir
# Number of files we are dealing with
!ls $file_dir | wc -l
# Get a python list of csv files
files = glob.glob(os.path.join(file_dir, "*.csv"))
# Look at a few to see how we can merge them
df1 = pd.read_csv(files[0])
df2 = pd.read_csv(files[1])
df3 = pd.read_csv(files[2])
df4 = pd.read_csv(files[3])
df5 = pd.read_csv(files[4])
df6 = pd.read_csv(files[5])
df7 = pd.read_csv(files[6])
df8 = pd.read_csv(files[7])
df9 = pd.read_csv(files[8])
df10 = pd.read_csv(files[9])
df11 = pd.read_csv(files[10])
df12 = pd.read_csv(files[11])
df13 = pd.read_csv(files[12])
print(df1.head(), "\n")
print(df2.head(), "\n")
print(df3.head(), "\n")
print(df4.head(), "\n")
print(df5.head(), "\n")
print(df6.head(), "\n")
print(df7.head(), "\n")
print(df8.head(), "\n")
print(df9.head(), "\n")
print(df10.head(), "\n")
print(df11.head(), "\n")
print(df12.head(), "\n")
print(df13.head(), "\n")

# Make a list of dataframes while adding a stick_ticker column
dataframes = [pd.read_csv(file).assign(stock_ticker=os.path.basename(file).strip(".csv")) for file in files]
# Concatenate all the dataframes into one
df = pd.concat(dataframes, ignore_index=True)
df.drop('stock_ticker',axis=1,inplace=True) 
df = df.replace(np.nan, '', regex=True)
#df.fillna(df.mean(), inplace=True)# fill missing values with mean column values
df["Date-Time_new"] = df["Date-Time"].astype(str) + df["timestamp"].astype(str)
df.drop('Date-Time',axis=1,inplace=True) 

df.drop('timestamp',axis=1,inplace=True) 


print(df.isnull().sum()) # count the number of NaN values in each column
df.head()
df.tail()
df.describe()
df.shape

type(df)
# Get a Series object containing the data type objects of each column of Dataframe.
# Index of series is column name.
dataTypeSeries = df.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)
X = df["Date-Time_new"] # Features
y = df["UV_Index"] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
X_train.head()
X_train.tail()
y_train.head()
print(X_train.dtype)
print(y_train.dtype)
print(X_test.dtype)
print(y_test.dtype)
X_train.describe
one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
x_train, x_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

