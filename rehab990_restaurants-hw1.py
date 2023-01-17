import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing Data (reading csv file)
df = pd.read_csv('../input/riyadhvenues/riyadh_venues.csv')
# Pandas Data Structures
df.head()
list(df.columns)
df.head(10)
df.shape
df.dtypes['users']
df.dtypes['rating']
df.tail(10)
df.loc[2:4,['name','categories','rating','users']]
df.loc[df['rating'] >9.0]
df.loc[(df['rating'] >9.0) & (df['price']=='cheap')]
df['categories']
df.groupby(['categories','price'])['users'].count()