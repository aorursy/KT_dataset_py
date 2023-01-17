# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/fifa19/data.csv")

data.head() #head show first 5 row
data.tail() # tail show last 5 row
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
print(data['Club'].value_counts())
data.describe().T
data.boxplot(column='Jumping',by='Balance')
data_new=data.head()

data_new
melted=pd.melt(frame=data_new,id_vars='Name',value_vars=['Composure','Balance'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1=data.head(5)

data2=data.tail(5)

data_new=pd.concat([data1,data2],axis=0)

data_new
data1=data["Overall"].head(5)

data2=data["Potential"].head(5)

data_new=pd.concat([data1,data2],axis=1)

data_new
data.dtypes
data['Photo']=data['Photo'].astype('category')

data.dtypes
data.isnull().values.any()
data.isnull().sum()
data["Release Clause"].fillna(0, inplace = True)
data.isnull().sum()