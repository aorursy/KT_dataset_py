import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
hu_df = pd.read_csv('/kaggle/input/lingtrain-hungarian-word-frequency/hu.csv')

hu_df['freq'] = pd.to_numeric(hu_df['freq'])
hu_df.head(10).plot(kind='pie',y='freq',subplots=True,figsize=(12,10),labels=hu_df.head(10)['word'])