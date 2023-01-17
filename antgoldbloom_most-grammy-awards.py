

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/grammy-awards/the_grammy_awards.csv')

df.groupby('artist')[['artist']].count().sort_values(ascending=False)[:20]