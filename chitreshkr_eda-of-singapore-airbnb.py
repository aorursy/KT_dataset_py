import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)
df.head()
df.describe()
df.info()
df.columns
df_null = df.loc[:, df.isnull().any()].isnull().sum().sort_values(ascending=False)

print(df_null)
correlation = df.corr(method='pearson')
correlation
f,ax = plt.subplots(figsize=(12,10))

sns.heatmap(df.iloc[:,2:].corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax,cmap="YlGnBu")



plt.show()
df.isna().sum()
sns.pairplot(df)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x=df['room_type'], y=df['price'])

plt.xticks(rotation= 45)

plt.xlabel('room_type')

plt.ylabel('Price')

plt.title('Price vs Room Type')
plt.figure(figsize=(25,10))

sns.barplot(x=df['neighbourhood_group'], y=df['price'])

plt.xlabel('neighbourhood_group')

plt.ylabel('Price')

plt.title('Price vs neighbourhood_group')
plt.figure(figsize=(25,10))

sns.barplot(x=df['neighbourhood'], y=df['price'])

plt.xticks(rotation= 90)

plt.xlabel('neighbourhood')

plt.ylabel('Price')

plt.title('Price vs neighbourhood')
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='availability_365',y='price',data=df,color='lime',alpha=0.8)

plt.xlabel('availability_365',fontsize = 15,color='blue')

plt.ylabel('price',fontsize = 15,color='blue')

plt.title('availability_365  VS  price',fontsize = 20,color='blue')

plt.grid()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='minimum_nights',y='price',data=df,color='lime',alpha=0.8)

plt.xlabel('minimum_nights',fontsize = 15,color='blue')

plt.ylabel('price',fontsize = 15,color='blue')

plt.title('minimum_nights  VS  price',fontsize = 20,color='blue')

plt.grid()