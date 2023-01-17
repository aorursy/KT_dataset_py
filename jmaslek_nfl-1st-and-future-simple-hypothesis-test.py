# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
playlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

injuries = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

trackdata = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
plt.figure(figsize=(5,3))

injuries['Surface'].value_counts().plot(kind='bar')

plt.title('Injuries By Surface')

plt.show()
import scipy

n_injuries = injuries.shape[0]

n_syn = injuries.Surface.value_counts()[0]

n_nat = injuries.Surface.value_counts()[1]

n_plays = playlist.PlayKey.nunique()

n_plays_syn = playlist.FieldType.value_counts()[1]

n_plays_nat = playlist.FieldType.value_counts()[0]



prop_syn = n_syn/n_plays_syn

print('Proportion of injuries on Synthetic field is ',prop_syn)

prop_nat = n_nat/n_plays_nat

print('Proportion of injuries on Natural field is ',prop_nat)



sigma_syn = np.sqrt(prop_syn*(1-prop_syn)/n_plays_syn)

#print('SD of Synthetic Field: ',sigma_syn)



sigma_nat = np.sqrt(prop_nat*(1-prop_nat)/n_plays_nat)

#print('SD of Natural Field: ',sigma_nat)



pooled_prop = (n_syn+n_nat)/(n_plays_nat+n_plays_syn)

pooled_SE = np.sqrt(pooled_prop*(1-pooled_prop)*(1/n_plays_syn + 1/n_plays_nat))



Z = (prop_syn - prop_nat)/pooled_SE

print('The Z score for the difference in proportions is :',Z)

p =  scipy.stats.norm.cdf(-abs(Z))*2 #twosided

p1 = 1-scipy.stats.norm.cdf(abs(Z))

print('Corresponding 2 tailed p value: ',p)

print('P Value for testing if p_syn>p_nat: ',p1)