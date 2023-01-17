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
df=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df
df.head()
#1.data gathering
#2.data assessing
#3.correct the incorrect datas[area_type,availability,total_sqft,bath,balcony]
#4.missing values[balcony,bath,society,size,location]
#5.cleaning data
sum(df['price'].isnull())
#data types 
df.info()
df.describe().T
df.sample(10)
#missing values
df.isnull().sum()
#handelling missing values
df.dropna(subset=['location'],inplace = True)   #forlocation

df['size']= df['size'].fillna('2 BHK')  #for size

df['society']= df['society'].fillna('No society')   #for society

df['bath']= df['bath'].fillna(df['bath'].min())    #for bath

df['balcony']= df['balcony'].fillna(df['bath'].min())   #for balcony
df.isnull().sum()
#correction of incorrect data types
#converting objects to category
df['area_type']=df['area_type'].astype('category')
df['availability']=df['availability'].astype('category')
df['society']=df['society'].astype('object')


df.info()
df.head()
#outlairs in size

df['size']=df['size'].replace(['2 Bedroom','3 Bedroom','4 Bedroom','5 Bedroom','6 Bedroom','1 Bedroom','8 Bedroom','7 Bedroom','9 Bedroom'],['2 BHK','3 BHK','4 BHK','5 BHK','6 BHK','1 BHK','8 BHK','7 BHK','9 BHK'])
df['size']=df['size'].replace(['1 RK','10 Bedroom','11 Bedroom','12 Bedroom','18 Bedroom','43 Bedroom'],['1 BHK','10 BHK','11 BHK','12 BHK','18 BHK','43 BHK'])
df['size'].value_counts()
#outlairs in availability

a=[s for s in df['availability'] if s not in ['Ready To Move','Immediate Possession']]
df['availability']=df['availability'].replace(a,'To be available soon')
df['availability'].value_counts()
#outliers in bath

df['bath']=df['bath'].replace([s for s in df['bath'] if s not in [1,2,3,4]],4)
df['bath'].value_counts()
#outliers in total_sqft

df['total_sqft']=df['total_sqft'].replace(['2100 - 2850'],750)
df['total_sqft']=df['total_sqft'].replace(['3010 - 3410'],750)
df['total_sqft']=df['total_sqft'].replace(['2957 - 3450'],750)
df['total_sqft']=df['total_sqft'].replace(['3067 - 8156'],750)
df['total_sqft'].value_counts()
df.head()
df.head(50)
