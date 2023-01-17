import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import seaborn as sns



# load museums data

museums = pd.read_csv('../input/museums.csv', low_memory=False)



# revenue columns contains some zeros and NaNs. lets get rid of them

museums = museums[~pd.isnull(museums['Revenue']) & (museums['Revenue'] > 0)]



# extract all zoos from the data

zoo_tag = 'ZOO, AQUARIUM, OR WILDLIFE CONSERVATION'

zoos = museums.loc[museums['Museum Type'].str.match(zoo_tag), :]



# and now extract all non-zoos from the data

non_zoos = museums.loc[~museums['Museum Type'].str.match(zoo_tag), :]
# plotting revenue data

ind = np.arange(2)

width = 0.5

plt.bar(ind, [zoos['Revenue'].mean(), non_zoos['Revenue'].mean()], width, tick_label=['ZOO', 'NON-ZOO'], color=['g', 'r'])

# is variance (STD) equal among the two groups?

equal_var = zoos['Revenue'].std() == non_zoos['Revenue'].std()

print("Do zoo and non-zoo revenue have equal variance (STD)? {}".format('Yes' if equal_var else 'No'))



# perform t-test

t, p = ttest_ind(zoos['Revenue'], non_zoos['Revenue'], equal_var=equal_var)

print("T-test confirms that zoos and non-zoos have significantly different average revenues (p={}).".format(p))
# histograms for each class' revenue distribution

fig, axes = plt.subplots(1, 2)



# data is highly skewed  

sns.distplot(zoos['Revenue'], kde=False, ax=axes[0]).set_title('Zoo Revenue Dist.')

sns.distplot(non_zoos['Revenue'], kde=False, ax=axes[1]).set_title('Non-Zoo Revenue Dist.')