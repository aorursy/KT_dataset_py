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
import matplotlib.pyplot as plt





Data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

Data.head()
Data.shape
Data.describe()
#Detect the outliers by using percentile

maxthresold=Data['price'].quantile(0.95)

maxthresold

#remove the outliers

Data[Data['price']>maxthresold]

Data.head()
#Detect the outliers 

minthresold=Data['price'].quantile(0.05)

minthresold
#Remove the outliers

Data[Data['price']<minthresold]

Data.head()
# Data has no outliers

Data_no_outliers=Data[(Data['price']<maxthresold) & (Data['price']>minthresold)]

Data_no_outliers.shape
Data_no_outliers.describe()