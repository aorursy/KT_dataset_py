# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import accumulate # iteration tool for cummulative summation

import matplotlib.pyplot as plt # data visualization package



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





df = pd.DataFrame.from_csv('../input/data.csv')

df.head()

# Any results you write to the current directory are saved as output.
year = df['YEAR']

s_yield  = df['SORGHUM-Yield/hectare (kg/Ha)']

s_area = df['SORGHUM-Area Planted (000 hect)']



cum_year = list(accumulate(year))

cum_yield = list(accumulate(s_yield))

cum_area = list(accumulate(s_area))



#ploting the data on a single axis



plt.plot(year, cum_yield, label='SORGHUM-Yield/')

plt.plot(year, cum_area, label='SORGHUM-Area')

plt.xlabel('Year')
fig = plt.figure()

ax1 = fig.add_subplot(111)



# Now defining the axis for one plot



ax2 = ax1.twinx()

lns2 = ax2.plot(year, cum_area, '-r', label='SORGHUM-Area')



plt.title("Intervention analysis and hypothesis Testing of Sorghum data")

lns1 = ax1.plot(year, cum_yield, '-b', label = 'SORGHUM-Yield')



# Combining axes lengends for both sides

lns = lns1

labs = [l.get_label() for l in lns]

ax1.legend(lns, labs, loc=0)



ax1.set_xlabel('Years', size=12)

# make the y-axis label

ax1.set_ylabel('SORGHUM-Yield/hectare (kg/Ha)', size=12)

ax1.tick_params('y')





ax1.legend(loc='best')

ax2.legend(loc='lower right')



ax2.set_ylabel("SORGHUM-Area Planted (000 hect)",color='r')

ax2.tick_params('y', colors='r')



ax1.grid()

ax2.grid()



fig.tight_layout()

plt.show() 