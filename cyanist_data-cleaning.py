import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/complete.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], low_memory=False)
# check if there are missing values in dataset

df.isnull().values.any()
# count NaN values in each column

df.isnull().sum()
# fill NaN values with '0'

scrubbed=df.fillna(value=0)

scrubbed
# check all columns' data types

scrubbed.dtypes
# change type of latitude to float

scrubbed['latitude'] = pd.to_numeric(scrubbed['latitude'],errors='coerce')
scrubbed.dtypes
# make city, state, country columns more pretty

scrubbed['city']=scrubbed['city'].str.title()

scrubbed['state']=scrubbed['state'].str.upper()

scrubbed['country']=scrubbed['country'].str.upper()
scrubbed.head()
# check if there are inappropraite values in dataset

scrubbed.describe().astype(np.int64).T
# replace inappropraite values with column mean

scrubbed.replace([97836000,0],scrubbed['duration (seconds)'].mean())