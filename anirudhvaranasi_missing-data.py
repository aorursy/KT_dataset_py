import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
d = {

      'A': [1, 2, np.nan],

      'B': [5, np.nan, np.nan],

      'C': [1, 2, 3]

     }

df = pd.DataFrame(d)

df
df.dropna(axis = 1)
df.dropna()
df.dropna(thresh=2)
df.fillna(value='FILL VALUE')
df['A'].fillna(value = df['A'].mean())