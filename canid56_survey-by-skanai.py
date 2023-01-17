import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset = pd.read_csv('../input/winemag-data-130k-v2.csv', sep=',', index_col=0)
dataset.iloc[:5]
dataset.dtypes
df_ctgr = dataset[dataset.columns[dataset.dtypes == 'object']].drop('description', axis=1)
df_ctgr.iloc[:5]
fig = plt.figure(figsize=(20,10))
i = 1
for col in df_ctgr:
    ax = plt.subplot(2,5,i)
    df_ctgr[col].value_counts().iloc[:10].plot.bar()
    (ymin, ymax) = ax.get_ylim()
    plt.text(9, 0.95*ymax,col, horizontalalignment='right')
    i += 1
plt.show()
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(1,1,1)
dataset.plot.scatter(x='price', y='points',ax=ax)
(ymin, ymax) = ax.get_ylim()
yticks = np.arange(np.ceil(ymin), np.floor(ymax)+1, 1)
plt.yticks(yticks)
plt.xscale('log')
plt.show()
print('hello')
