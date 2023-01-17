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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df.shape
df.head()
df2 = df.drop(['area_type','availability','society','balcony'],axis = 1)
df2.head()
df2.isnull().sum()
df3 = df2.dropna()
df3.isnull().sum()
df3['size'].unique()
df3['BHK'] = df3["size"].apply(lambda x:int(x.split(" ")[0]))
df3.head()
df3['BHK'].unique()
df3[df3.BHK > 20]
df3.total_sqft.unique()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df3[~df3['total_sqft'].apply(is_float)].head()
def conv_sq_to_m(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0])+ float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(conv_sq_to_m)
df4.head()
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 1000000/df5['total_sqft']
df5.head()
df5.location.unique().shape
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats
loc_less_than_10 = location_stats[location_stats <= 10 ]
loc_less_than_10
len(df5.location.unique())
df5.location = df5.location.apply(lambda x: 'other' if x in loc_less_than_10 else x)
len(df5.location.unique())
df5[df5.total_sqft/df5.BHK <300].head()
df5.shape
df6= df5[~(df5.total_sqft/df5.BHK <300)]
df6.shape
df6.price_per_sqft.describe()
def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_outliers(df6)
df7.shape
        

