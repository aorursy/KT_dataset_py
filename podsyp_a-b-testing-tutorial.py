# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import statsmodels.api as sm

from tqdm.autonotebook import tqdm

from matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/how-to-do-product-analytics/product.csv', sep=',')

df.shape
df.head()
df.groupby('site_version')[['user_id']].count()
df.target.mean()
df.groupby('site_version').target.mean()
# Creating an list with bootstrapped means for each AB-group

boot_1d = []

for i in tqdm(range(100), leave=False):

    boot_mean = df.sample(frac=1, replace=True).groupby('site_version').target.mean()

    boot_1d.append(boot_mean)

    

# Transforming the list to a DataFrame

boot_1d = pd.DataFrame(boot_1d)

    

# A Kernel Density Estimate plot of the bootstrap distributions

boot_1d.plot.kde();
boot_1d.head()
# Adding a column with the % difference between the two AB-groups

boot_1d['diff'] = (boot_1d.desktop-boot_1d.mobile)/boot_1d.mobile * 100



# Ploting the bootstrap % difference

ax = boot_1d['diff'].plot.kde()
# Calculating the probability that 1-day retention is greater when the gate is at level 30

prob = (boot_1d['diff']>0).mean()



# Pretty printing the probability

print('{:.1%}'.format(prob))
# Calculating 7-day retention for both AB-groups

df.groupby('site_version').target.mean()
n_rows_desktop, n_rows_mobile = df.groupby('site_version').target.count()

trgt_desktop, trgt_mobile = df.groupby('site_version').target.mean()
n_rows_desktop, n_rows_mobile, trgt_desktop, trgt_mobile
interval = 1.96 * np.sqrt((trgt_desktop*(1-trgt_desktop)/n_rows_desktop) + (trgt_mobile*(1-trgt_mobile)/n_rows_mobile))

print("Lower bond: {0:.3f}%, upper bond: {1:.3f}%".format(trgt_desktop-trgt_mobile-interval, trgt_desktop-trgt_mobile+interval))
z_score, p_value = sm.stats.proportions_ztest([trgt_mobile*n_rows_mobile, trgt_desktop*n_rows_desktop], 

                                              [n_rows_mobile, n_rows_desktop], 

                                              alternative='smaller')

print("Z-score={0:.3f},  p_value={1:.3f}".format(z_score, p_value))