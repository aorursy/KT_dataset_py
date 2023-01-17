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
df= pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df.head()
#Maintain a copy of your data
df_copy= df.copy()
#1. Asking Questions
#what coloumn will contribute in my anaylsis
# what columns are not useful
# Data Preprocessing
# 1. Gathering data[Done]
# 2. Assessing Data
#-----a. Incorrect data types[area_type,bath,balcony,price ]
#---->b. Missing values [location,balcony,bath,society,size]
#---->c. removing outliers in bath col
# 3. Cleaning data
# shape of data
df.shape
# Data type of cols
df.info()
# Check for missing values
df.isnull().sum()
# Mathemetical cols
df.describe().T
# Handling Missing values

#fillna
df['balcony']= df['balcony'].fillna(df['balcony'].mean())
df['bath']= df['bath'].fillna(df['bath'].mean())
df['society'] = df['society'].fillna('Not Available')
df['size'] = df['size'].fillna('Not Available')
#dropna
df.dropna(subset=['location'],inplace=True)
#handling incorrect data type
df['area_type']= df['area_type'].astype('category')
df['total_sqft']= df['total_sqft'].astype('object')
df['bath']= df['bath'].astype('float32')
df['balcony']= df['balcony'].astype('float32')
df['price']= df['price'].astype('float32')
#handling outlier data of bath
mask1=df['bath'] <= 20
df=df[mask1]
df.isnull().sum()
df.info()
