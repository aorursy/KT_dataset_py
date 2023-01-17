# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import statsmodels.formula.api as sm

sns.set(font_scale=1.5)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
flights = pd.read_csv('../input/flights.csv',dtype={'ORIGIN_AIRPORT': np.str, 'DESTINATION_AIRPORT': np.str})
print(flights.columns)
# convert SCHEDULED_DEPARTURE which is HHMM to seconds since midnight

flights['SCHEDULED_DEPARTURE_MINUTES'] =  (flights.SCHEDULED_DEPARTURE // 100) * 60 + flights.SCHEDULED_DEPARTURE % 100
flights['SCHEDULED_DEPARTURE_15MIN'] = flights.SCHEDULED_DEPARTURE_MINUTES // 15 * 15

flights['SCHEDULED_DEPARTURE_HOUR'] = flights.SCHEDULED_DEPARTURE_MINUTES // 60
print(flights.columns)
# remove cancelled and diverted flights

fset = flights[(flights.CANCELLED == 0) & (flights.DIVERTED == 0)]

fset.ARRIVAL_DELAY.describe()
fset.ARRIVAL_DELAY.mode()
dims = (15,5)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='firebrick')

sns.boxplot(x='ARRIVAL_DELAY', data=fset,meanprops=meanpointprops, showmeans=True)
fset.hist(column='ARRIVAL_DELAY',bins=100)

fset.DISTANCE.describe()
fset.DISTANCE.mode()
sns.boxplot(x='DISTANCE', data=fset,meanprops=meanpointprops, showmeans=True)
fset.hist(column='DISTANCE',bins=100)
result = sm.ols(formula="ARRIVAL_DELAY ~ DISTANCE", data=fset).fit()

print(result.summary())


    

# takes a long time to run

#sns.jointplot(fset['DISTANCE'],fset['ARRIVAL_DELAY'], kind="reg")

fset.SCHEDULED_DEPARTURE_MINUTES.describe()
fset.SCHEDULED_DEPARTURE_MINUTES.mode()
sns.boxplot(x='SCHEDULED_DEPARTURE_MINUTES', data=fset,meanprops=meanpointprops, showmeans=True)
fset.hist(column='SCHEDULED_DEPARTURE_MINUTES')
result = sm.ols(formula="ARRIVAL_DELAY ~ SCHEDULED_DEPARTURE_MINUTES", data=fset).fit()

print(result.summary())
# takes a long time to run

# sns.jointplot(fset['SCHEDULED_DEPARTURE_MINUTES'],fset['ARRIVAL_DELAY'], kind="reg")
dims = (20,10)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

chart = sns.boxplot(y='ARRIVAL_DELAY',x='SCHEDULED_DEPARTURE_15MIN', data=fset, showfliers=False,meanprops=meanpointprops, showmeans=True)

ax = plt.gca()

def fmt(x,v):

    return int(x * 15)

#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=4))

for item in chart.get_xticklabels():

    item.set_rotation(60)

plt.show()
fset.AIRLINE.describe()
dims = (10,5)

fig, ax = plt.subplots(figsize=dims)

sns.countplot(x='AIRLINE', data=fset)
dims = (10,5)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='firebrick')

sns.boxplot(y='ARRIVAL_DELAY',x='AIRLINE', data=fset,meanprops=meanpointprops, showmeans=True)
dims = (10,5)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

sns.boxplot(y='ARRIVAL_DELAY',x='AIRLINE', data=fset, showfliers=False, meanprops=meanpointprops, showmeans=True)
fset.ORIGIN_AIRPORT.describe()
dims = (20,8)

fig, ax = plt.subplots(figsize=dims)

chart = sns.countplot(x='ORIGIN_AIRPORT',data=fset, order=pd.value_counts(fset['ORIGIN_AIRPORT']).iloc[:75].index)

for item in chart.get_xticklabels():

    item.set_rotation(60)
sns.set(font_scale=1.5)

dims = (20,8)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

chart = sns.boxplot(y='ARRIVAL_DELAY',x='ORIGIN_AIRPORT', data=fset, order=pd.value_counts(fset['ORIGIN_AIRPORT']).iloc[:40].index, showfliers=False, meanprops=meanpointprops, showmeans=True)

for item in chart.get_xticklabels():

    item.set_rotation(60)
# analyze for three airports best: 10397, mid: SEA, worst: ORD

sample = fset[fset.ORIGIN_AIRPORT.isin(['10397','SEA','ORD'])]
gr = sample[(sample.SCHEDULED_DEPARTURE_HOUR >= 5) & (sample.SCHEDULED_DEPARTURE_HOUR < 24)]

dims = (20,10)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

fig, ax = plt.subplots(figsize=dims)

chart = sns.boxplot(y='ARRIVAL_DELAY',x='SCHEDULED_DEPARTURE_HOUR',hue='ORIGIN_AIRPORT', data=gr, showfliers=False, showmeans=True, meanprops=meanpointprops)

for item in chart.get_xticklabels():

    item.set_rotation(60)
# analyze for three airlines best: DL, mid: UA, worst: NK

sample2 = fset[fset.AIRLINE.isin(['DL','UA','NK'])]
gr = sample2[(sample2.SCHEDULED_DEPARTURE_HOUR >= 5) & (sample2.SCHEDULED_DEPARTURE_HOUR < 24)]

dims = (20,10)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

fig, ax = plt.subplots(figsize=dims)

chart = sns.boxplot(y='ARRIVAL_DELAY',x='SCHEDULED_DEPARTURE_HOUR',hue='AIRLINE', data=gr, showfliers=False, showmeans=True, meanprops=meanpointprops)

for item in chart.get_xticklabels():

    item.set_rotation(60)
# fist 20 Airports

dims = (20,8)

fig, ax = plt.subplots(figsize=dims)

meanpointprops = dict(marker="s",markeredgecolor='black',markerfacecolor='black')

chart = sns.boxplot(y='ARRIVAL_DELAY',x='ORIGIN_AIRPORT',hue='AIRLINE', data=sample2, order=pd.value_counts(sample2['ORIGIN_AIRPORT']).iloc[:20].index, showfliers=False, showmeans=True, meanprops=meanpointprops)

for item in chart.get_xticklabels():

    item.set_rotation(60)