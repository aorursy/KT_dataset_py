import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing Data (reading csv file)

df = pd.read_csv('../input/riyadhvenues/riyadh_venues.csv')
# Pandas Data Structures

df.head()
df.columns
df.head(10)
df.shape
# r=df['rating'].dtypes

# u= df['users'].dtypes

# print('rating types is : ', r ,'user types is : ', u)

df.dtypes[4:6]
df.tail(10)
df.loc[2:4, ['name','categories','rating','users']]
df[df.rating>9.0].head()
ch= df[df['price']=="Cheap"]

ch[df['rating']>9.0]
df.groupby('categories').count()
df[['categories','price','users']].groupby(['categories','price']).sum()