from scipy.stats import ttest_ind # just the t-test from scipy.stats

from scipy.stats import probplot # for a qqplot

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math 

data = pd.read_csv('../input/museums.csv')

data.describe()
histmuseum = data['Museum Type'] == 'HISTORY MUSEUM'

genmuseum = data['Museum Type'] == 'GENERAL MUSEUM'

# hr = history museum revenue

hr = data[histmuseum]['Revenue']

# gr = general museum revenue

gr = data[genmuseum]['Revenue']



hr.head()
# removing duplicates d=no dulicates

dhr = hr.drop_duplicates()

dgr = gr.drop_duplicates()

# removing NAN values n=no NaN

ndhr = dhr.dropna()

ndgr = dgr.dropna()
fig = plt.figure(figsize=(10 , 4))

hist = ndgr

# Histogram

plt.hist(hist, bins=10, color='grey', edgecolor='white')

#Customizations

plt.xlabel('')

plt.ylabel('')

plt.title('')

plt.show()

plt.clf()
z_score = (ndgr - ndgr.mean())/ndgr.std(ddof=0)

hist = z_score

# Histogram

plt.hist(hist, bins=10, color='grey', edgecolor='white')

#Customizations

plt.xlabel('')

plt.ylabel('')

plt.title('')

# Show and clear plot

plt.show()

plt.clf()

# appling log to both series

logndhr = ndhr.apply(np.log)

logndgr = ndgr.apply(np.log)

# getting rid of all the inf

noinfndhr = logndhr > 0

noinfndgr = logndgr > 0

#finalized verison for both types

finHistory = logndhr[noinfndhr]

finGeneral = logndgr[noinfndgr]
plt.figure(figsize=(14 , 7))

sns.distplot(finHistory)

sns.distplot(finGeneral)
probplot(finHistory, dist="norm", plot=plt)

probplot(finGeneral, dist="norm", plot=plt)

ttest_ind(finHistory, finGeneral, equal_var=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))



# Plotting the first histogram.

ax[0].hist(finHistory, edgecolor="white")

ax[0].axvline(finHistory.mean(), c="r")  # Plotting the mean value.

ax[0].set_title("Revenue distribution of History Museum's")



# Plotting the second histogram.

ax[1].hist(finGeneral, edgecolor="white")

ax[1].axvline(finGeneral.mean(), c="r")  # Plotting the mean value.

ax[1].set_title("Revenue distribution of General Museum's")