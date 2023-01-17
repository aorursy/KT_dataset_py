import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

p = sns.color_palette()
with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
print('Number of rows: {}, Number of columns: {}'.format(*df.shape))
cols = [0, 0, 0]

for c in df.columns:

    if 'derived' in c: cols[0] += 1

    if 'fundamental' in c: cols[1] += 1

    if 'technical' in c: cols[2] += 1

print('Derived columns: {}, Fundamental columns: {}, Technical columns: {}'.format(*cols))

print('\nColumn dtypes:')

print(df.dtypes.value_counts())

print('\nint16 columns:')

print(df.columns[df.dtypes == 'int16'])
df.describe()
#cor = df.corr()

mask = np.zeros_like(cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
from xgboost import XGBRegressor
model = XGBRegressor()

model.fit(df.drop(['y'], axis = 1, inplace = False), df.y)
model.predict(df.drop(['y'], axis = 1, inplace = False))
