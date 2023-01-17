import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
path = '../input/airsense_data/node_data.csv'

TextFileReader = pd.read_csv(path, chunksize=10000)  # the number of rows per chunk



dfList = []

for df in TextFileReader:

    dfList.append(df)



df = pd.concat(dfList,sort=False)

del dfList

gc.collect()
df.shape
df.head()