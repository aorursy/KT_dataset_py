%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)
frame1 = pd.read_csv('../input/train.csv')

frame1.head()
frame1.describe()
frame1.info()
test = pd.read_csv('../input/test.csv')

test.head()
sns.countplot(x="target", data = frame1)
sns.barplot(x="ps_ind_02_cat", y="target", data=frame1, estimator=sum)
missing_values = []

missing = []

for f in frame1.columns:

    miss = frame1[frame1[f] == -1][f].count()

    if miss > 0:

        missing_values.append(f)

        missing_perc = miss/frame1.shape[0]

        missing.append(missing_perc*100)

#        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, miss, missing_perc))

D = dict(zip(missing_values, missing))



plt.bar(range(len(D)), D.values())

plt.xticks(range(len(D)), D.keys(), rotation = 'vertical')



plt.show()
del frame1['id']

del frame1['target']
ps_bin = []

ps_cat = []

ps_cont = []

for i in frame1.columns:

    if i.endswith("bin"):

        a = frame1.groupby(i).size()

        ps_bin.append(a)

    elif i.endswith("cat"):

        b = frame1.groupby(i).size()

        ps_cat.append(b)

    else:

        c = frame1.groupby(i).size()

        ps_cont.append(c)
df = pd.concat(ps_bin, axis=0, keys=[i.index.name for i in ps_bin]).unstack()

df.plot(kind='bar', stacked=True)
df = pd.concat(ps_cat, axis=0, keys=[i.index.name for i in ps_cat]).unstack()

df.plot(kind='bar', stacked=True, legend=False)