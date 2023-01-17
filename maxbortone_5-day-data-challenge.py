import numpy as np # linear algebra

import matplotlib.pyplot as plt # plotting

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

df = pd.read_csv('../input/20170308hundehalter.csv')

# describe only non-numerical data

summary1 = df.describe(exclude=[np.number])

summary1
# describe only numerical data

df2 = df[df.columns.difference(['HALTER_ID', 'RASSE2_MISCHLING', 'STADTKREIS', 'STADTQUARTIER'])]

summary2 = df2.describe(exclude=[np.object])

summary2
# describe the frequency of the city districts (STADTKREIS)

s = df.iloc[:, 3]

s.value_counts()
# plot histogram of city districts (STADTKREIS)

ax = s.hist(bins=11, grid=False, width=0.5)

ax.set_title('City districts')

ax.set_ylabel('counts')

ax.set_xlabel('districts')

ax.set_xticks(np.arange(1, 12)+0.25)

ax.set_xticklabels(np.arange(1, 12))
from scipy.stats import ttest_ind

# import the two datasets

df2015 = pd.read_csv('../input/20151001hundehalter.csv')

df2016 = pd.read_csv('../input/20160307hundehalter.csv')



# select district column and get counts

districts2015 = df2015.iloc[:, 3]

districts2016 = df2016.iloc[:, 3]

count2015 = districts2015.value_counts()

count2016 = districts2016.value_counts()

print("Standard deviations for the two datasets: \n\t- 2015: {:.3f}\n\t- 2016: {:.3f}"

      .format(count2015.std(), count2016.std()))



# perform t-test

t, prob = ttest_ind(count2015, count2016, equal_var=False)

print("t-test results: \n\t- t = {:.3f} \n\t- p = {:.3f}".format(t, prob))



# plot histograms

f, ax = plt.subplots(1, 1, figsize=(11, 8))

ax.hist([districts2015, districts2016], bins=11, range=(1, 12))

ax.set_title('City districts')

ax.set_ylabel('counts')

ax.set_xlabel('districts')

ax.set_xticks(np.arange(1, 12)+0.5)

ax.set_xticklabels(np.arange(1, 12))