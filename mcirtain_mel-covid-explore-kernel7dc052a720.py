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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
df.describe() # investigate the max deaths--where, when...
# most deaths reported in Italy, 3/22/20 (5,476)

df[df['Deaths'] == 5476] 
# Total global deaths as of 3/22:  14,634

df[df['ObservationDate'] == '03/22/2020']['Deaths'].sum() 
# Reported cases globally as of March 8 (14 days prior to 3/22):  109,835

df[df['ObservationDate'] == '03/08/2020']['Confirmed'].sum()
# Reported cases in Italy as of March 8: 7,375

df[(df['ObservationDate'] == '03/08/2020') & (df['Country/Region'] == 'Italy')]



# Reported cases in Italy as of March 12: 12,462

df[(df['ObservationDate'] == '03/12/2020') & (df['Country/Region'] == 'Italy')]

# Recoveries in Italy as of 3/22: 7,024

# Deaths in Italy as of 3/22: 5,476

df[(df['ObservationDate'] == '03/22/2020') & (df['Country/Region'] == 'Italy')]
# Get death rate estimates for varying 'average days between diagnosis and death'

from datetime import datetime as dt



def get_dates_in_range(num_days=14):

    '''

    return list of date strings from 'today' going back specified number of days

    '''

    if num_days < 1:

        num_days = 1

    dates = []

    for days in range(1, num_days):

        today = dt.now().date()

        todays_date = today.strftime('%m/%d/%Y')

        month = today.strftime('%m')

        day = today.strftime('%d')

        year = today.strftime('%Y')

        newday = int(day) - days

        date_of_interest = f'{month}/{newday}/{year}'

        dates.append(date_of_interest)

    return dates



def get_stats(date='', day_range=0, country=''):

    if date == '':

        date = today.strftime('%m/%d/%Y')

    if country == '':

        country = 'Italy'

    dates = get_dates_in_range(day_range)

    my_data = df[(df['Country/Region'] == country) & (df['ObservationDate'] == date)]

    for date in dates:

        print(date)

    return my_data
