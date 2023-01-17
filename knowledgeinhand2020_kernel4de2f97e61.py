# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
m1=pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
m1.head()

m1.shape
m1.groupby('area_type')['area_type'].agg('count')
m1.isnull().sum()
m2=m1.drop(['area_type','availability','society','balcony'], axis ='columns')
m2.isnull().sum()
m3=m2.dropna()
m3.isnull().sum()
m3.shape
m3['size'].unique()
m3['bhk']=m3['size'].apply(lambda x:int (x.split(' ')[0]))
m3.head()
m3['bhk'].unique()
m3['location'].unique()
m3[m3.bhk>20]
m3['total_sqft'].unique
def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
m3[~m3['total_sqft'].apply(is_float).head(10)]