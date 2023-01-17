# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.dates import WeekdayLocator, DateFormatter

from datetime import timedelta

%matplotlib inline
covid19 = pd.read_csv('/kaggle/input/hospital-resources-during-covid19-pandemic/Hospitalization_all_locs.csv', parse_dates=['date'], 

                    usecols=['location_name','date', 'allbed_mean', 'ICUbed_mean', 'InvVen_mean'])
covid19.rename(columns={'location_name':'state'}, inplace=True)



# create a list of the states you want data for, as you can use this list later on

states_list = ['New York', 'Louisiana', 'Washington', 'California', 'Alabama']

covid19 = covid19[covid19['state'].isin(states_list)].copy()
covid19['Resources'] = covid19.loc[:, ['allbed_mean', 'ICUbed_mean', 'InvVen_mean']].sum(axis=1).div(1000)
fig, ax = plt.subplots(figsize=(20,10))



for st in states_list:

    ax.plot(covid19[covid19.state == st].date,covid19[covid19.state == st].Resources,label = st)

ax.xaxis.set_major_locator(WeekdayLocator())

ax.xaxis.set_major_formatter(DateFormatter("%b %d"))

min_date = covid19.date[covid19.Resources!=0].min().date() - timedelta(days=7)

max_date = covid19.date[covid19.Resources!=0].max().date() + timedelta(days=7)

ax.set_xlim(min_date, max_date)

fig.autofmt_xdate()



ax.legend()



font_size = 14

plt.title('The hospital resources needed for COVID-19 patients across 5 different US States', fontsize=font_size+2)

plt.ylabel("Total Resource Count (k)", fontsize=font_size)

plt.xlabel("Date", fontsize=font_size)



plt.show()

fig.savefig('Hospital_resource_use.png', bbox_inches='tight')