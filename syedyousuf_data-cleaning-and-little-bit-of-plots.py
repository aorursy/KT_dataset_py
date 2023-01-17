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
df=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.head()
df.dtypes
df.drop(index=10472,inplace=True)
def removing_postfix_and_prefix(column):

    column=column.str.replace('+','')

    column=column.str.replace(',','')

    column=column.str.replace('$','')

    column=column.str.replace("''","")

    return column
df['Installs']=removing_postfix_and_prefix(df['Installs'])

df['Price']=removing_postfix_and_prefix(df['Price'])
def change_dtype(column):

    column=column.astype('float')

    return column
df['Price']=change_dtype(df['Price'])

df['Installs']=change_dtype(df['Installs'])

df['Reviews']=change_dtype(df['Reviews'])
def remove_na(df):

    return df.dropna(inpalce=True)
def fill_na(df,methods):

    return df.fillna(method=methods,inplace=True)
fill_na(df,'ffill')
def change_to_date_time(column):

    column=pd.to_datetime(column)

    return column
df['Last Updated']=change_to_date_time(df['Last Updated'])
df.dtypes
import seaborn as sns

import matplotlib.pyplot as plt
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(25,10))

plt.suptitle('Count plots')

sns.countplot(y='Category',data=df,ax=ax1)

sns.countplot('Type',data=df,ax=ax2)

sns.countplot('Content Rating',data=df,ax=ax3)

plt.show()