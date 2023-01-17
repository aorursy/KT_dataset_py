import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('../input/train.csv')
for c in train.columns[2:]:

        stats = train.groupby([c])['y'].mean().reset_index().rename(columns={c: c,

                                                                             'y': c+'_mean'})

        x = train.merge(stats, on=c, how='left')

        train[c+'_mean'] = x[c+'_mean']

        train.drop(c,inplace=True,axis=1)
remove = []

c = train.columns

for i in range(len(c)):

    v = train[c[i]].values

    for j in range(i+1, len(c)):

        if np.array_equal(v, train[c[j]].values):

            remove.append(c[j])
dupes = []

for r in remove:

    dupes.append(r.replace('_mean',''))

dupes = np.unique(dupes)

dupes = sorted(dupes)
print(dupes)