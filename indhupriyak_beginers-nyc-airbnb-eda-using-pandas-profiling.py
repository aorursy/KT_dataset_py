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
#load datasets
train=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

train.head(5)
train.shape
train.info()
train.describe()
train.isnull().sum()
train['reviews_per_month']=train['reviews_per_month'].fillna(0.0)
train.drop(['id','host_name','last_review'],axis=1,inplace=True)
train.drop(['name'],axis=1,inplace=True)
train.head(5)
train.isna().sum()
#to get a simple report with pandas profiling
from pandas_profiling import ProfileReport
report = ProfileReport(train, title='Pandas Profiling Report')
#to create a report use to_widgets
report.to_widgets()
report.to_file('output.html')
train.dtypes
#convert to category dtype
train['neighbourhood_group'] = train['neighbourhood_group'].astype('category')
#use .cat.codes to create new colums with encoded value
train['neighbourhood_group_cat'] = train['neighbourhood_group'].cat.codes
train.head(5)
#convert to category dtype
train['room_type'] = train['room_type'].astype('category')
#use .cat.codes to create new colums with encoded value
train['room_type_cat'] = train['room_type'].cat.codes
train.head(1)
#convert to category dtype
train['neighbourhood'] = train['neighbourhood'].astype('category')
#use .cat.codes to create new colums with encoded value
train['neighbourhood_cat'] = train['neighbourhood'].cat.codes
train.head(1)