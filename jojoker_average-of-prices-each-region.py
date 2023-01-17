import numpy as np

import pandas as pd

import seaborn as sns

sns.set()
raw_data = pd.read_csv('../input/singapore-airbnb/listings.csv')

raw_data.info()
sns.boxplot(x=raw_data['minimum_nights'])
sns.boxplot(x=raw_data['price'])
def remove_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr #formula for find the outlier on the left side

    fence_high = q3+1.5*iqr #formula for find the outlier on the right side

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out
data_remove_1 = remove_outlier(raw_data,'minimum_nights')

data_remove_2 = remove_outlier(data_remove_1,'price')
sns.distplot(data_remove_1['minimum_nights'], bins=10, kde=False)
sns.distplot(data_remove_2['price'], bins=10, kde=False)
data = data_remove_2[['neighbourhood_group','price']]
data['neighbourhood_group'].value_counts()
data.groupby('neighbourhood_group')['price'].mean()
data['price'].mean()