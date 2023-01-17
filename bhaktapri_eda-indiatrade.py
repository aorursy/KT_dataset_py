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
import_df= pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")

export_df= pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
export_df.head()
import_df.head()
export_df['value'].describe() 
import_df['value'].describe()
print("total null values in export data:",export_df['value'].isnull().sum())

print("total null values in export data:",import_df['value'].isnull().sum())
def filling_null(data_df):

    data_df["value"].fillna(data_df['value'].mean(),inplace = True)

    import_df.year = pd.Categorical(import_df.year)

    return data_df
import_df = filling_null(import_df)

export_df = filling_null(export_df)

## Import And Export Country Wise

import_df1= import_df.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)

export_df1= export_df.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)
import_df1=import_df1.head(10)

export_df1=export_df1.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')


plt.figure(figsize=(10,6))

sns.barplot(import_df1.index,import_df1.value)

plt.xticks(rotation=35)
plt.figure(figsize=(10,6))

sns.barplot(export_df1.index,export_df1.value)

plt.xticks(rotation=35)
import_Commodity= import_df.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending = False)

export_Commodity= export_df.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending = False)

import_Commodity=import_Commodity.head(10);

export_Commodity=export_Commodity.head(10);
plt.figure(figsize=(10,10))

sns.barplot(import_Commodity.value,import_Commodity.index,palette = 'Set1')
plt.figure(figsize=(10,10))

sns.barplot(export_Commodity.value,export_Commodity.index,palette='tab20')
import_yearly=import_df.groupby('year').agg({'value':'sum'})

export_yearly=export_df.groupby('year').agg({'value':'sum'})
plt.figure(figsize=(10,6))

sns.barplot(import_yearly.index,import_yearly.value,palette='plasma')
plt.figure(figsize=(10,6))

sns.barplot(export_yearly.index,export_yearly.value,palette='cividis')
plt.figure(figsize= (18,10))

sns.lineplot(x='year',y='value', data=import_df, label='Imports')

sns.lineplot(x='year',y='value', data=export_df, label='Exports')

plt.title('Values of Indian imports and exports', fontsize=16)

plt.xlabel('Year')

plt.ylabel('Value in million US$')

plt.show()