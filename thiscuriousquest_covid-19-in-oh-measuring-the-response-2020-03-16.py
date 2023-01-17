import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [18,10]
from datetime import timedelta, date



def daterange(date1, date2):

    for n in range(int ((date2 - date1).days)+1):

        yield date1 + timedelta(n)



start_dt = date(2020, 1, 22)

end_dt = date.today() - timedelta(days=1)

date_list = [dt.strftime("%m-%d-%Y") for dt in daterange(start_dt, end_dt)]

# works on local machine, won't work in kaggle notebook

# Note: time zone has to be adjusted

list_of_dfs = [pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(i)) for i in date_list]
#dfs = pd.concat(list_of_dfs)
df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-16-2020.csv')
hubei = df.loc[df['Province/State'] == 'Hubei']

italy = df.loc[df['Country/Region'] == 'Italy']

ohio = df.loc[df['Province/State'] == 'Ohio']
three_regions = pd.concat([hubei,italy,ohio])

three_regions.at[1, 'Province/State'] = 'Italy'

three_regions.drop(['Country/Region', 'Latitude', 'Longitude'], axis=1, inplace=True)

three_regions.reset_index(drop=True, inplace=True)

three_regions
three_regions.plot(kind='bar', x='Province/State', y=['Confirmed','Deaths','Recovered'])

plt.ylabel('Thousands of People')

plt.show()