

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

years = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]

data = pd.read_csv('../input/arrests.csv')

data.head()

data.dtypes

data_sum = data.loc[data.Sector=='All'].sum()



count = data_sum.values[-34:]

total = np.array(count[::2])

mexicans = np.array(count[1::2])

other = total-mexicans

print(other, mexicans, other)

plt.bar(years, mexicans, label='mexicans')

plt.bar(years, other, bottom=mexicans, color="red", label='other')

plt.xticks(rotation=50)

plt.ylabel('Amount')

plt.title('Illegal Immigrants of the USA')

plt.legend()

plt.show()
#Percent Mexican



per_mexicans = np.array([x[0]/x[1] for x in zip(mexicans,total)])

per_other = np.ones(len(per_mexicans))-per_mexicans



plt.bar(years, per_mexicans, label='mexicans')

plt.bar(years, per_other, bottom=per_mexicans, color="red", label='other')

plt.xticks(rotation=50)

plt.ylabel('Amount')

plt.title('Percentage of Illegal Immigrants of the USA')

plt.legend(bbox_to_anchor=[1.3,1.0])

plt.show()
from scipy import optimize 



def polynom(x, a, b):

    return  a*np.exp((x-2000)*b)



mexicans = np.array(mexicans).astype(dtype='float64')

popt, kl = optimize.curve_fit(polynom, years, mexicans)



x = np.arange(2000,2021,0.5)

y = [polynom(xi, *popt) for xi in x]

plt.bar(years, mexicans, label='mexicans')

plt.bar(x[-8::2], y[-8::2], color='orange')

plt.plot(x,y, color='orange')

plt.xlim([2000,2021])

plt.ylim([0,2000000])

plt.xticks(rotation=50)

plt.show()
list(zip(x[-8::2],y[-8::2]))
wall_immigrants = data.loc[(data.Border=='Southwest') & (data.Sector == 'All')]

north_immigrants = data.loc[(data.Border=='North') & (data.Sector == 'All')]

coast_immigrants = data.loc[(data.Border=='Coast') & (data.Sector == 'All')]



wall_count = wall_immigrants.values[0][3:]

wall_total = np.array(wall_count[::2])

#mexicans = np.array(count[1::2])

#other = total-mexicans



north_count = north_immigrants.values[0][3:]

north_total = np.array(north_count[::2])



coast_count = coast_immigrants.values[0][3:]

coast_total = np.array(coast_count[::2])



plt.bar(years, wall_total, label='Southwest', color='red')

plt.bar(years, coast_total, bottom=wall_total, color='blue', label='Coast')

plt.bar(years, north_total, bottom=coast_total+wall_total, color='green', label='North')

plt.xticks(rotation=50)

plt.ylabel('Amount')

plt.title('Illegal Immigrants of the USA')

plt.legend()

plt.show()