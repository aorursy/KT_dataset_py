import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing Data (reading csv file)
df = pd.read_csv('../input/riyadhvenues/riyadh_venues.csv')
# Pandas Data Structures
df.head()
list (df.columns)
df.loc[0:9]
df.shape
df.dtypes [['rating','users']]

df.tail(10)
df.loc[2:4, ['name', 'categories','rating','users' ] ]
df[ (df.rating > 9.0) ]
df [(df['rating'] >9.0) & (df['price'] == 'Cheap') 
 ]
df['categories']
df.groupby(['categories', 'price'])['users'].count()