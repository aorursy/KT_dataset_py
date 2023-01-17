# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/used-cars/vehicle.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
sns.set(color_codes=True)
from sklearn import datasets , linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
from sklearn.metrics import accuracy_score
df=pd.read_csv('/kaggle/input/used-cars/vehicles.csv')
df.head(10) #starting 10 rows of the dataset
df.tail()
df.info() #getting the summary of the dataframe
df.shape #for  summary of the data frame
df.describe(include='all')
df.isnull().sum() #checking if there are some null values
df = df.drop(["id","url","region_url","drive","paint_color","image_url","description","county","vin","transmission","lat","long"], axis=1) 
#these are the colomn to be drop which meant to no use to me .
df.head(5)
df.head(10)
df.isnull().sum() #checking null values after dropping the coloumn  .
df.shape #Total number of rows and columns
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
df = df.drop_duplicates()
df.head(5)
df.shape #Total number of rows and columnsafter dropping the values
df = df.dropna()
df.count()
sns.boxplot(x=df['price'])
sns.boxplot(x=df[''])