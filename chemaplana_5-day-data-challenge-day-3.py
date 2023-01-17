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
dfc = pd.read_csv('../input/cereal.csv')

print (dfc.info())
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
dfc_hot = dfc[dfc['type'] == 'H']

dfc_cold = dfc[dfc['type'] == 'C']
stats.ttest_ind(dfc_hot['sugars'], dfc_cold['sugars'], equal_var=False)
stats.ttest_ind(dfc_hot['sodium'], dfc_cold['sodium'], equal_var=False)
stats.ttest_ind(dfc_hot['sodium'], dfc_hot['sodium'], equal_var=False)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

sns.distplot(dfc_hot['sugars'], kde=False, ax=axes[0], color='b', label = 'hot')

sns.distplot(dfc_cold['sugars'], kde=False, ax=axes[0], color='g', label = 'cold')

sns.distplot(dfc_hot['sodium'], kde=False, ax=axes[1], color='b', label = 'hot')

sns.distplot(dfc_cold['sodium'], kde=False, ax=axes[1], color='g', label = 'cold')

axes[0].legend()

axes[1].legend()
print (dfc['rating'].describe())
dfc_hi = dfc[dfc['rating'] > 40]

dfc_lo = dfc[dfc['rating'] <= 40]

for i in pd.Series(dfc.columns[3:]):

    print (i, stats.ttest_ind(dfc_hi[i], dfc_lo[i], equal_var=False))
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

sns.distplot(dfc_hi['shelf'], kde=False, ax=axes[0], color='b', label = 'hi')

sns.distplot(dfc_lo['shelf'], kde=False, ax=axes[0], color='g', label = 'low')

sns.distplot(dfc_hi['sugars'], kde=False, ax=axes[1], color='b', label = 'hi')

sns.distplot(dfc_lo['sugars'], kde=False, ax=axes[1], color='g', label = 'low')

axes[0].legend()

axes[1].legend()