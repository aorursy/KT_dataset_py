# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mp

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



scrubbed = pd.read_csv('../input/ufo-sightings/scrubbed.csv')

pop = pd.read_csv('../input/united-nations-world-populations/UNpopfile.csv')

sns.set(style='white', context='notebook', palette='deep')



#ca_pop.head()

# Any results you write to the current directory are saved as output.
import datetime

import time

year_occurance = []

year_report =  []

for time in scrubbed['datetime']:

    if time[-5:] != '24:00':

        year_occurance.append(datetime.datetime.strptime(time, "%m/%d/%Y %H:%M").year)

for time in scrubbed['date posted']:

    year_report.append(datetime.datetime.strptime(time, "%m/%d/%Y").year)



plt.figure(1)

plt.hist(year_report, bins = 50, range = (1940,2015))

plt.hist(year_occurance, bins = 50, range = (1940,2015))

plt.ylabel('Amount')

plt.xlabel('Year')

#plt.figure(2)

#plt.ylabel('Number of reports')

#plt.xlabel('Year')

#plt.hist(year_report, bins = 30, range = (1940,2015))

year_pop = []

year = []

for i in range(40000):

    if pop['Location'][i] == 'United States of America':

        year_pop.append(pop['PopTotal'][i])

        year.append(pop['Time'][i])

year_pop = year_pop[:65]

year = year[:65]

year_pop_fitted = []

for p in range(65):

    year_pop_fitted.append((year_pop[p])*0.0533333333)

plt.plot(year, year_pop)

plt.ylabel('1000 people')

plt.xlabel('Year')
plt.figure(1)

plt.plot(year, year_pop_fitted,color = 'r')

plt.hist(year_report, bins = 50, range = (1940,2015))

plt.hist(year_occurance, bins = 50, range = (1940,2015))
plt.hist(year_report, bins = 10, range = (2005,2015))

plt.hist(year_occurance, bins = 10, range = (2005,2015))

plt.ylabel('Amount')

plt.xlabel('Year')
year_pop_fitted_2 = []

for p in range(60):

    year_pop_fitted_2.append((year_pop[p])*0.0266666667)

year_pop_fitted_2 = year_pop_fitted_2[-10:]

plt.plot(year[-10:], year_pop_fitted_2,color = 'r')

plt.hist(year_report, bins = 11, range = (2005,2015))

plt.hist(year_occurance, bins = 11, range = (2005,2015))

plt.ylabel('Amount')

plt.xlabel('Year')
year_pop_fitted_3 = []

for p in range(55):

    year_pop_fitted_3.append((year_pop[p])*0.0266666667)

year_pop_fitted_3 = year_pop_fitted_3[-7:]

plt.plot(year[-10:][:7], year_pop_fitted_3,color = 'r')

plt.hist(year_report, bins = 7, range = (2005,2011))

plt.hist(year_occurance, bins = 7, range = (2005,2011))

plt.ylabel('Amount')

plt.xlabel('Year')