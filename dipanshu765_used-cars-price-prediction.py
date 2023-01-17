import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
original_data = df
df.shape
df.isnull().sum()
df = df.drop(['county'], axis=1)
df.columns
df = df.dropna(subset=['size', 'odometer', 'cylinders', 'paint_color', 'lat'])
df = df.drop(['vin', 'condition', 'drive', 'manufacturer'], axis=1)
df.isnull().sum()
df.info()
df = df.drop(['id', 'url'], axis=1)
print(df.nunique())
cols_to_drop = ['region', 'region_url', 'model', 'image_url', 'description', 'state']
df = df.drop(cols_to_drop, axis=1)
df2 = df
df=pd.get_dummies(data=df, 
               columns=[
                       'cylinders',
                        'fuel', 
                        'title_status', 
                        'transmission', 
                        'size', 
                        'type', 
                        'paint_color'
               ], 
               drop_first=True)
imputer = KNNImputer(n_neighbors=2)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.corr()['price']
