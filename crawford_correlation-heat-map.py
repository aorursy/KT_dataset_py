import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
df = pd.read_csv("../input/WNBA Stats.csv")
df.head()
t = pd.DataFrame(data={"col": df.dtypes.index, "type": df.dtypes}).reset_index(drop=True)

col_names = t["col"][t.type != "object"]

col_names
fig = plt.figure(figsize=(12,12))



ax = fig.add_subplot(1,1,1)

cax = ax.matshow(df.corr(), interpolation = 'nearest')



fig.colorbar(cax)



ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



ax.set_xticklabels(col_names)

ax.set_yticklabels(col_names);
