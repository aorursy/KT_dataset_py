# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

ds = pd.read_csv("../input/database.csv")



ds.describe()
ds.Magnitude.hist()

plt.show()
# plot locations

plt.scatter(ds.Longitude, ds.Latitude, s=(ds.Magnitude-5.5)*20, alpha=0.5)

plt.show()
m = ds.Depth.mean()

s = ds.Depth.std()

print('mean=',m,', stdev=',s)
ds.Depth.hist()

plt.show()
# plot empirical cdf

import statsmodels.api as sm



ecdf = sm.distributions.ECDF(ds.Depth)



x = np.linspace(min(ds.Depth), max(ds.Depth))

y = ecdf(x)

plt.step(x, y)

plt.grid()

plt.show()

# handle dates

import datetime



# date is stored as string!

test = ds.Date[212]

print(test)

# convert to datetime object

test_date = datetime.datetime.strptime(test, "%m/%d/%Y")

print(test_date)



# for whole data frame use pandas

dates = pd.to_datetime(ds.Date)



dates.describe()
dates_num = pd.to_numeric(dates) / 1e9 / 60 / 60 / 24 / 365.25

dates_num.describe()

plt.scatter(dates_num, ds.Magnitude, marker=".")

plt.show()
# select only subset

ds_big = ds[ds.Magnitude>8]

# remove some columns

ds_big = ds_big.drop('Depth Error',1)

ds_big = ds_big.drop('Depth Seismic Stations',1)

ds_big = ds_big.drop('Magnitude Error',1)

ds_big = ds_big.drop('Magnitude Seismic Stations',1)

ds_big = ds_big.drop('Horizontal Error',1)

ds_big = ds_big.drop('Root Mean Square',1)

# summary

ds_big.describe()
# plot magnitude vs time

dates_big_num = pd.to_numeric(pd.to_datetime(ds_big.Date)) / 1e9 / 60 / 60 / 24 / 365.25

plt.scatter(dates_big_num, ds_big.Magnitude, marker="D")

plt.grid()

plt.show()
# select only a subset of columns

ds_big_sel = ds_big[['Date','Magnitude','Depth']]

print(ds_big_sel)
# sort by size

ds_big_sel_sort = ds_big_sel.sort_values(by=['Magnitude'], ascending=False) # sort

ds_big_sel_sort = ds_big_sel_sort.reset_index(drop=True) # adjust index to new sorting

print(ds_big_sel_sort)

plt.plot(ds_big_sel_sort.Magnitude)

plt.grid()

plt.show()
# depth vs magnitude

plt.scatter(ds_big_sel_sort.Magnitude, ds_big_sel_sort.Depth)

plt.grid()

plt.show()
# plot locations of big EQs

plt.scatter(ds_big.Longitude, ds_big.Latitude, s=(ds_big.Magnitude-8)*1000, c=ds_big.Depth, alpha=0.5)

plt.grid()

plt.show()