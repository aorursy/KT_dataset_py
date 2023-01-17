# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)
frame2 = pd.read_csv('../input/train.csv')

frame1 = pd.DataFrame(frame2)

del frame1['id']

del frame1['target']
missing_values = []

missing = []

for f in frame1.columns:

    miss = frame1[frame1[f] == -1][f].count()

    if miss > 0:

        missing_values.append(f)

        missing_perc = miss/frame1.shape[0]

        missing.append(missing_perc*100)

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, miss, missing_perc))

D = dict(zip(missing_values, missing))



plt.bar(range(len(D)), D.values())

plt.xticks(range(len(D)), D.keys(), rotation = 'vertical')



plt.show()
ps_bin = []

ps_cat = []

ps_cont = []

for i in frame1.columns:

    if i.endswith("bin"):

        x = frame1.groupby(i).size()

        ps_bin.append(x)

    elif i.endswith("cat"):

        y = frame1.groupby(i).size()

        ps_cat.append(y)

    else:

        z = frame1.groupby(i).size()

        ps_cont.append(z)
bin_df = pd.concat(ps_bin, axis=0, keys=[s.index.name for s in ps_bin]).unstack()

bin_df.plot(kind='bar', stacked=True, grid=False, figsize=(10, 8))

bin_df = pd.concat(ps_cat, axis=0, keys=[s.index.name for s in ps_cat]).unstack()

bin_df.plot(kind='bar', stacked=True, grid=False, figsize=(10, 8), legend=False)