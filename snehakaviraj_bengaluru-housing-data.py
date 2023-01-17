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
df = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
#df.head()
df.sample(5)
# Asking Questions
# 1. Which columns are important for my analysis? [All]
# 2. Which columns can be dropped? [None]

# Data Preprocessing
# 1. Gathering data [Done]
# 2. Assessing data
# ------> a) Missing values in [location, size, society, bath, balcony]
# ------> b) Incorrect datatypes [area_type, location, ]
# Shape of data
df.shape
# Data type of cols.
df.info()
df.isnull().sum()
# Maintaining a copy of data
df_copy = df.copy()
df.describe().T
df['size'].value_counts()
df['society'].value_counts()
df['bath'].value_counts()
df['balcony'].value_counts()
# 3. Cleaning Data
# ----> a) Handling missing values:

#location
df['location'] = df['location'].fillna('No Location')
# size
df['size'] = df['size'].fillna("2 BHK")
df['size'] = df['size'].str.replace("Bedroom","BHK", case=False)
df['size'] = df['size'].str.extract(r'([\d:,'']+)')
# society
df['society'] = df['society'].fillna('Other')
#bath
df['bath'] = df['bath'].fillna('2')
# balcony
df['balcony'] = df['balcony'].fillna(df['balcony'].mean())
df.info()
# ----> b) Handling incorrect datatype:

#area_type [category]
df['area_type'] = df['area_type'].astype('category')

#availability [category]
df['availability'] = df['availability'].astype('category')

#location [category]
df['location'] = df['location'].astype('category')

#size [int]
df['size'] = df['size'].astype('int32')

#society [category]
df['society'] = df['society'].astype('category')

#total_sqft [category] 
df['total_sqft'] = df['total_sqft'].astype('category')

#bath [int]
df['bath'] = df['bath'].astype('int32')

#balcony[int]
df['balcony'] = df['balcony'].astype('int32')

#price [float]
df['price'] = df['price'].astype('float32')

