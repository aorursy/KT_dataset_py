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
import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/Electronics_IT_Exports_1.csv")

df.head()
df.isnull().sum()
df.describe()
df.info()
df['2011-12 (Estimated)'].fillna(0,inplace = True)
df.head()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(df.Item,palette="muted")

plt.show()


df.plot(kind='scatter', x='2000-01', y='2010-11')



# axis labels

plt.xlabel("electronic it export")

plt.ylabel("electronic it export 2000-1 to 2010-11")



# axis limits

plt.xlim(20, 50)

plt.ylim(15, 55)
# function has it's own defination

def check_null_or_valid(row_data):

    '''Function that takes a row of data, 

       drops off all missing value and then 

       validate whether all values are greater 

       than 0 or not.'''

    row_without_na = row_data.dropna()

    row_numeric = pd.to_numeric(row_without_na)

    greater_than_0 = row_numeric >= 0

    return greater_than_0



# verify that first column is 'Item' 

assert df.columns[0] == 'Item'



# verify all values in columns are valid or not [axis = 1 (row-wise)]

assert df.iloc[:,1:].apply(check_null_or_valid, axis=1).all().all()



# verify that there is only one instance of each Item

assert df.Item.value_counts()[0] == 1
df_melt = pd.melt(df, id_vars='Item', var_name='year', value_name='life_expectancy')

df_melt.shape
df_melt.head()


pattern = '^[A-Za-z\.\s()\',-]*$'



Item = df_melt.Item





Item = Item.drop_duplicates()





mask = Item.str.contains(pattern)





mask_inverse = ~mask





invalid_Item_names = Item[mask_inverse]

invalid_Item_names



assert pd.notnull(df_melt.Item).all()



assert pd.notnull(df_melt.year).all()
df_melt = df_melt.dropna()

df_melt.shape
# grouping by 'year' and aggregate 'life_expectancy' by the mean

df_melt_agg = df_melt.groupby('year').life_expectancy.mean()



# Line plot of life expectancy per year

df_melt_agg.plot()



# Add title and specify axis labels

plt.title('Life expectancy over the years')

plt.ylabel('Life expectancy')

plt.xlabel('year')