# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
temp = pd.read_excel('../input/temp.xls')
temp = temp.set_index ('YEAR')
temp.head()
temp.describe()
temp['ANNUAL'].idxmax()
x = temp.index
y = temp.ANNUAL

plt.scatter(x,y)
plt.show()
mean_months = temp.loc[:,'JAN':'DEC'].mean()
print (mean_months)
plt.plot(mean_months.index, mean_months)
hottest_seasons = {'Winter' : temp['JAN-FEB'].idxmax(),
                   'Summer' : temp['MAR-MAY'].idxmax(),
                   'Monsoon': temp['JUN-SEP'].idxmax(),
                   'Autumn' : temp['OCT-DEC'].idxmax()}
print (hottest_seasons)
temp ['DIFF'] = temp.loc[:,'JAN':'DEC'].max(axis=1) - temp.loc[:,'JAN':'DEC'].min(axis=1)
temp.DIFF.idxmax()
axes= plt.axes()
axes.set_ylim([5,15])
axes.set_xlim([1901,2017])
plt.plot(temp.index, temp.DIFF)

temp.DIFF.mean()
year_dict = temp.loc[:,'JAN':'DEC'].to_dict(orient='index')

sorted_months = []
for key, value in year_dict.items():
    sorted_months.append (sorted(value, key=value.get)[:4]) #Only take first 4 elements out
winter = sorted_months[:]
winter_set = []
for x in winter:
    winter_set.append (set(x))
temp['WINTER'] = winter_set
winter_routine = max(sorted_months, key=sorted_months.count)
temp.WINTER [temp.WINTER != set(winter_routine)]
