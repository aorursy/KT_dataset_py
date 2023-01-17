import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
df
df.dropna()
df.dropna(axis=1)
df.dropna(thresh=2)
df.fillna(value='FILL VALUE')
df['A'].fillna(value=df['A'].mean())