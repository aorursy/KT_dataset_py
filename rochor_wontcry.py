import pandas as pd
import numpy as np
df = pd.read_csv('/kaggle/input/dataport-export_gas_oct2015-mar2016.csv')
df.shape
df.head()
df['dataid'] = df['dataid'].astype(str)
df.info()
houses = df['dataid'].unique()
print('Number of houses (number of unique meters) is ' + str(len(houses)))
df['dataid'].describe()
df[df['dataid']=='739']