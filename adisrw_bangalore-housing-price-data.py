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
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
df=pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df.head()
df.shape
df.groupby('area_type')['area_type'].agg('count')
df1=df.drop(['area_type','availability','society','balcony'], axis='columns')
df1.head()
df1.isnull().sum()
df2=df1.dropna()
df2.isnull().sum()
df2.shape
df2['size'].unique()
df2['bhk']=df2['size'].apply(lambda x: int(x.split(' ')[0]))
df2.head()
df2['bhk'].unique()
df2[df2.bhk>20]
df2.total_sqft.unique()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df2[~df2['total_sqft'].apply(is_float)].head(10)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
convert_sqft_to_num('3090 - 5002')
df3=df2.copy()
df3['total_sqft']=df3['total_sqft'].apply(convert_sqft_to_num)
df3.head()
df3.loc[30]
dfs= df3.copy()
dfs['price_per_sqft']= dfs['price']*100000/dfs['total_sqft']
dfs.head()
len(dfs.location.unique())
dfs.location = dfs.location.apply(lambda x : x.strip())
location_stats=dfs.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
len(location_stats[location_stats<=10])
location_stats_less_than_10= location_stats[location_stats<=10]
location_stats_less_than_10
len(dfs.location.unique())
dfs.location=dfs.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(dfs.location.unique())
len(dfs.location.unique())
dfs.head(10)
dfs[dfs.total_sqft/dfs.bhk<300].head()
dfs.head(10)
dfs[dfs.total_sqft/dfs.bhk<300].head()
dfs.shape
df5=dfs[~(dfs.total_sqft/dfs.bhk<300)]
df5.shape
df5.price_per_sqft.describe()
def remove_pps_outliers(df):
    df_out =pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft>(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6=remove_pps_outliers(df5)
df6.shape
