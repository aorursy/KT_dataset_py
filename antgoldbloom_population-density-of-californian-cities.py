import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/california-cities/california_cities.csv',index_col=[0])

df['AreaInt'] = df['Area'].str.extract('([0-9]+)').astype(int)

df['AreaInt'][df.index == 'San Francisco'] = 47

(df['Population']/df['AreaInt']).sort_values(ascending=False)