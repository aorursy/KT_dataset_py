# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/winemag-data_first150k.csv')

df.head()



# Any results you write to the current directory are saved as output.
df.info()
# We remove the null entries

df=df[(df['price'].notnull()) & (df['country'].notnull())]

df.info()
df['country'].value_counts().head()
# We want to see if the distributions of wine prices vary for wines from the US, 

# as opposed to wines from Italy, France and Spain combined. 



price_us = df[df['country']=='US']['price']

price_itfrasp = df[(df['country']=='Italy') | (df['country']=='France') | (df['country']=='Spain')]['price']
print("The number of wines from the US are {} while the wines from Italy, France and Spain are {}".format(price_us.shape[0],price_itfrasp.shape[0]))
# Do the t-test for the two independent samples, assume unequal population variances

ttest_ind(price_us,price_itfrasp,equal_var=False)
stats.describe(price_us)
stats.describe(price_itfrasp)
# Plot the two distributions

# For ease of representation, we set xlim at 200

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6),sharex=True)

ax1.hist(price_us,bins=500)

ax2.hist(price_itfrasp,bins=500)

ax1.set_xlim(0,200)

ax2.set_xlim(0,200)

ax1.set_title('Price distributions of US made wines')

ax2.set_title('Price distributions of wines made in Italy, France and Spain');

fig.text(0.5, 0.04, 'Price', ha='center');

# We see that the distributions look different, which is reasonable

# considering the results of the t-test
