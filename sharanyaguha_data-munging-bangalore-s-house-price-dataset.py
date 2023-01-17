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
# Handling Missing values
# Changing the incorrect data type
# Maintaining a copy of data
# Removing outliers
# Displaying PRICE properly, with proper decimal points
df.head()
df.shape
# Handling Missing values

df['balcony'] = df['balcony'].fillna(0)
df['society'] = df['society'].fillna('Not Available')
df['location'] = df['location'].fillna('Sarjapur  Road')
df['bath'] = df['bath'].fillna(df['bath'].mean())
df['size'] = df['size'].fillna('3 BHK')
df.isnull().sum()
# Changing the incorrect data type 

df['bath']=df['bath'].astype('int32')
df['balcony']=df['balcony'].astype('int32')
df['price']=df['price'].astype('float32')
df['area_type']=df['area_type'].astype('category')
df['availability']=df['availability'].astype('category')
df['size']=df['size'].astype('category')
df.info()
# Maintaining a copy of data
df_copy=df.copy()
# Removing outliers in 'SIZE'

df['size']=df['size'].replace('1 Bedroom','1 BHK')
df['size']=df['size'].replace('2 Bedroom','2 BHK')
df['size']=df['size'].replace('3 Bedroom','3 BHK')
df['size']=df['size'].replace('4 Bedroom','4 BHK')
df['size']=df['size'].replace('5 Bedroom','5 BHK')
df['size']=df['size'].replace('6 Bedroom','6 BHK')
df['size']=df['size'].replace([s for s in df['size'] if s not in ['1 BHK','2 BHK','3 BHK','4 BHK','5 BHK', '6 BHK']],'3 BHK')
df['size'].value_counts()
# Removing outliers in 'AVAILABILITY'

abc=[s for s in df['availability'] if s not in ['Ready To Move','Immediate Possession']]
df['availability']=df['availability'].replace(abc,'To be available soon')
df['availability'].value_counts()
# Removing outliers in 'BATH'

df['bath']=df['bath'].replace([s for s in df['bath'] if s not in [1,2,3,4]],4)
df['bath'].value_counts()
# Renaming location to 'Others' where there are less than 10 houses

g=df.groupby('location').groups
abc=[x for x in g if len(g[x]) <10]
df['location']=df['location'].replace(abc,'others')
# Renaming society to 'Others' where there are less than 5 houses

g=df.groupby('society').groups
abc=[x for x in g if len(g[x]) <5]
df['society']=df['society'].replace(abc,'others')
# Displaying PRICE properly, with proper decimal points

df['price']=df['price'].apply(lambda x: x*1000)
df['price']=df['price'].apply(lambda x: round(x,2))
df['price']=df['price'].apply(lambda x: format(x,".2f"))
df.head(20)