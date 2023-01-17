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
# import liberaries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import matplotlib.gridspec as gridspec

from matplotlib import rcParams

import seaborn as sns



rcParams.update({'figure.autolayout': True})



%matplotlib notebook



plt.style.use('fivethirtyeight')
## import covid-19 datasets

df_covid = pd.read_csv("/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv")

df_covid.head()



##import states population

df_population = pd.read_csv("http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#")

df_population = df_population[['NAME', 'POPESTIMATE2019']].iloc[5:].rename(columns = {'NAME': 'state name','POPESTIMATE2019':'population'})

df_covid.head()



## replace state abbv with full name

df_population['state'] = df_population['state name'].map({

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

})

df_population.head()
df_covid.columns
## clean data and change column name

my_data = df_covid.join(df_population.set_index('state'), on = 'state')

my_data[my_data['population'].notnull()]

data_cleaned = my_data[my_data['population'].notnull()][['date', 'state name', 'population', 'positive', 'death',

                                                    'totalTestResults', 'deathIncrease', 'positiveIncrease', 'totalTestResultsIncrease']].rename(columns = {

   'state name':'state', 'positive': 'total_case', 'death':'total_death', 'totalTestResults' : 'total_test', 'deathIncrease' : 'new_death', 'positiveIncrease': 'new_case', 'totalTestResultsIncrease': 'new_test'

})

data_cleaned['date'] = pd.to_datetime(data_cleaned['date'], format='%Y%m%d')

data_cleaned = data_cleaned[data_cleaned['total_case']>0 ]



## add metrics per million people

data_cleaned['case_per_m'] = data_cleaned['total_case']*1000000/data_cleaned['population']

data_cleaned['death_per_m'] = data_cleaned['total_death']*1000000/data_cleaned['population']

data_cleaned['test_per_m'] = data_cleaned['total_test']*1000000/data_cleaned['population']

data_cleaned.head()
top_10_case = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('total_case', ascending = True).iloc[-10:].set_index('state')['total_case']

top_10_death = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('total_death', ascending = True).iloc[-10:].set_index('state')['total_death']

top_10_test = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('total_test', ascending = True).iloc[-10:].set_index('state')['total_test']

top_10_case_per_m = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('case_per_m', ascending = True).iloc[-10:].set_index('state')['case_per_m']

top_10_death_per_m = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('death_per_m', ascending = True).iloc[-10:].set_index('state')['death_per_m']

top_10_test_per_m = data_cleaned[data_cleaned['date'] == '2020-05-10'].sort_values('test_per_m', ascending = True).iloc[-10:].set_index('state')['test_per_m']

plt.figure(figsize=(12,10))



ax1 = plt.subplot(3,2,1)

ax1.barh(top_10_case.index,top_10_case.values)

ax1.tick_params(size = 5,labelsize = 12)

plt.xlabel("Confirmed Cases",fontsize=10, axes = ax1)

plt.title("Confirmed Cases",fontsize=12, axes = ax1)

ax1.grid(alpha=0.5)



ax2 = plt.subplot(3,2,2)

ax2.barh(top_10_case_per_m.index,top_10_case_per_m.values)

ax2.tick_params(size = 5,labelsize = 12)

plt.xlabel("Confirmed Cases per million",fontsize=10, axes = ax2)

plt.title("Confirmed Cases per million",fontsize=12, axes = ax2)

ax2.grid(alpha=0.5)



ax3 = plt.subplot(3,2,3)

ax3.barh(top_10_death.index,top_10_death.values, color = 'grey')

ax3.tick_params(size = 5,labelsize = 12)

plt.xlabel("Deaths",fontsize=10, axes = ax3)

plt.title("Deaths",fontsize=12, axes = ax3)

ax3.grid(alpha=0.5)



ax4 = plt.subplot(3,2,4)

ax4.barh(top_10_death_per_m.index,top_10_death_per_m.values, color = 'grey')

ax4.tick_params(size = 5,labelsize = 12)

plt.xlabel("Deaths per million",fontsize=10, axes = ax4)

plt.title("Deaths per million",fontsize=12, axes = ax4)

ax4.grid(alpha=0.5)



ax5 = plt.subplot(3,2,5)

ax5.barh(top_10_test.index,top_10_test.values, color = 'orange')

ax5.tick_params(size = 5,labelsize = 12)

plt.xlabel("Tests",fontsize=10, axes = ax5)

plt.title("Tests",fontsize=12, axes = ax5)

ax5.grid(alpha=0.5)



ax6 = plt.subplot(3,2,6)

ax6.barh(top_10_test_per_m.index,top_10_test_per_m.values, color = 'orange')

ax6.tick_params(size = 5,labelsize = 12)

plt.xlabel("Tests per million",fontsize=10, axes = ax6)

plt.title("Tests  per million",fontsize=12, axes = ax6)

ax6.grid(alpha=0.5)

plt.show()
data_tracker = data_cleaned[data_cleaned['total_case']>=100]

data_tracker.head()
data_tracker['days'] = data_tracker.groupby('state')['date'].apply(lambda x: x - min(x)).dt.days
data_tracker['days']

fig = plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(111)



ax1.plot(data_tracker[data_tracker['state'] == 'New York']["days"], data_tracker[data_tracker['state'] == 'New York']["total_case"],'-')

ax1.plot(data_tracker[data_tracker['state'] == 'California']["days"], data_tracker[data_tracker['state'] == 'California']["total_case"],'-')

ax1.plot(data_tracker[data_tracker['state'] == 'Florida']["days"], data_tracker[data_tracker['state'] == 'Florida']["total_case"],'-')

ax1.plot(data_tracker[data_tracker['state'] == 'Texas']["days"], data_tracker[data_tracker['state'] == 'Texas']["total_case"],'-')

ax1.plot(data_tracker[data_tracker['state'] == 'Pennsylvania']["days"], data_tracker[data_tracker['state'] == 'Pennsylvania']["total_case"],'-')

ax1.set_yscale('log')

from matplotlib.ticker import ScalarFormatter

for axis in [ax1.xaxis, ax1.yaxis]:

    axis.set_major_formatter(ScalarFormatter())

ax1.legend(['New York', 'California', 'Florida', 'Texas', 'Pennsylvania'])

ax1.set_xlabel('Number of days since 100 cases')

ax1.set_ylabel('Confirmed cases')

ax1.set_title('Confirmed cases')

plt.show()