import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing Data (reading csv file)
df = pd.read_csv('../input/riyadhvenues/riyadh_venues.csv')
# Pandas Data Structures
df.head()
df.columns
df.head(10)
df.shape
df.dtypes[4:6]

df.tail(10)
df.iloc[2:5 ,[0,1,4,5]]
df[df["rating"]>9]
R=df[df["rating"]>9]
R[df["price"]=="Cheap"]

v=df[['categories','name']]
v.groupby('categories').count()

v=df[['categories','price','users']]
v.groupby(['price','categories']).sum()