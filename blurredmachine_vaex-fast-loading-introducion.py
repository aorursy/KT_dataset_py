!pip install vaex==2.5.0 
import vaex



import pandas as pd

import numpy as np
n_rows = 100000 # one hundred thousand random data

n_cols = 10

df = pd.DataFrame(np.random.randint(0, 100, size=(n_rows, n_cols)), columns=['c%d' % i for i in range(n_cols)])

df.head()
df.info(memory_usage='deep')
file_path = 'main_dataset.csv'

df.to_csv(file_path, index=False)
vaex_df = vaex.from_csv(file_path)
type(vaex_df)
import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
vaex_df = vaex.open('/kaggle/working/main_dataset.csv')
type(vaex_df)
vaex_df.head()
%%time

vaex_df['multiplication_col13']=vaex_df.c1*vaex_df.c3
vaex_df['multiplication_col13']
vaex_df[vaex_df.c2>70]
dff=vaex_df[vaex_df.c2>70]
dff.c2.minmax(progress='widget')