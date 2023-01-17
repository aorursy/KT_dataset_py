import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv', index_col='row_id')
place_counts = train.place_id.value_counts()
mask = (place_counts[train.place_id.values] >= 8).values
train = train.loc[mask]

g = train.groupby('place_id')
mu = g[['x','y']].mean()
sigma = g[['x','y']].std()