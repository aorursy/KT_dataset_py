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
import matplotlib.pyplot as plt
df=pd.read_csv('../input/fortune1000/fortune1000.csv',index_col='Rank')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df['Sector'].values
df.index
df['Sector'].dtype
df['Revenue'].sum()
df['Employees'].mean()
df.head(2)
df.tail(2)
df[df['Sector']== 'Industrials']
df[df['Sector']!= 'Industrials']
df[df['Revenue']>50000]
sectors=df.groupby("Sector")

sectors
type(sectors)
sectors
len(sectors)
df.nunique()
df['Sector'].nunique()
sectors.size().sort_values(ascending=False).plot(kind='bar');
sectors.first()
sectors.last()
sectors.groups
df.loc[24]
sectors.get_group('Energy')
sectors.max()
sectors.min()
sectors.sum()
sectors.mean()
sectors['Employees'].sum()
sectors[['Profits','Employees']].sum()
sector=df.groupby(['Sector','Industry'])

sector.size()
sector.sum()
sector['Revenue'].sum()
sector['Employees'].mean()
sectors.agg({'Revenue':'sum','Profits':'sum','Employees':'mean'})
sectors.agg(['size','sum','mean'])
df1=pd.DataFrame(columns=df.columns)

df1
for sector, data in sectors:

    highest_revenue_company_in_group=data.nlargest(1,'Revenue')

    df1=df1.append(highest_revenue_company_in_group)

df1    
cities=df.groupby('Location')

df2=pd.DataFrame(columns=df.columns)

df2

for city, data in cities:

    highest_revenue_company_in_group=data.nlargest(1,'Revenue')

    df2=df2.append(highest_revenue_company_in_group)

df2
df[df['Revenue'].between(25000,50000)]
pd.get_option('precision')
pd.get_option('precision',2)
pd.reset_option('precision')
pd.get_option('precision')
pd.options.display.max_rows
pd.options.display.max_rows=4
df
pd.get_option('max_rows')
pd.get_option('max_columns')
pd.set_option('max_rows',6)

df
pd.set_option('max_columns',3)

df
pd.pandas.set_option('display.max_columns',None)

df