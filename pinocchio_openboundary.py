import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,5)
train = pd.read_csv('../input/train.csv', index_col='row_id')
train[train.x<0.1].x.hist(bins=100)

plt.xlabel('x')
plt.ylabel('Number of checkins');