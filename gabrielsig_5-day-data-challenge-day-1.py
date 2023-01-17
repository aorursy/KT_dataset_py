import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/archive.csv')
df.dropna(inplace=True)

df.head(10)
df.describe()
df.info()