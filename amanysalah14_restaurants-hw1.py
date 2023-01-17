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
X=df[['name','categories','rating']]
X[2:5]
df[df["rating"]>9]

Y=df[df["price"]=="Cheap"]
Y[df["rating"]>9]
A=df[['categories','name']]
A.groupby('categories').count()
a=df[['categories','price','users']]
a.groupby(['price','categories']).sum()