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
import seaborn as sns
import matplotlib.pyplot as plt
d=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
d
# 1. Asking Questions
# 2. Data Preprocessing
# 3. EDA
# 4. Drawing Conclusions
# 5. Communicating
# Asking Questions

# 1. What cols will contribute in my analysis -all 
# 2. What cols are not useful-none
# Data Preprocessing

# 1. Gathering Data [Done]
# 2. Assessing data
# ------ a. Incorrect data types [bath,balcony,size,price,availability,area_type]
# ------ b. Missing values in [Location,size,society,bath,balcony]
# 3. Cleaning Data
# make a copy of original 

dc=d.copy()
dc
dc.isnull().sum()
dc.info()
# let make all missing into normal one
dc['bath'] = dc['bath'].fillna(dc['bath'].mean())
dc['balcony'] = dc['balcony'].fillna(dc['balcony'].mean())
dc['society'] = dc['society'].fillna('No Specify')
dc.dropna(subset=['location'],inplace=True)

dc['size'] = dc['size'].fillna('2 BHK')     
# 2 BHK beacause maximum number is 2BHK houses
# let make all incorrect data types into correct ones.
dc['bath'] = dc['bath'].astype('int32')
dc['balcony'] = dc['balcony'].astype('int32')
dc['price'] = dc['price'].astype('int32')
dc['size'] = dc['size'].astype('category')
dc['availability'] = dc['availability'].astype('category')
dc['area_type'] = dc['area_type'].astype('category')
# now check
dc.info()
#let observe(obsevation krte hai)
dc
sns.pairplot(dc)
dc.describe().T
dc['area_type'].value_counts()

