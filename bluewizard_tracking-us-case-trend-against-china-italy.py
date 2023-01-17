import numpy as np

import pandas as pd

from datetime import datetime, timedelta

from matplotlib import pyplot as plt



df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
set(df['Country/Region'])
countries = ['US', 'Mainland China', 'Italy']



for country in countries:

    tmp = df[df['Country/Region'] == country].groupby(['ObservationDate']).sum()

    plt.plot(tmp.Confirmed)
print(df[df['Country/Region'] == 'US'].groupby(['ObservationDate']).sum())

print(df[df['Country/Region'] == 'Mainland China'].groupby(['ObservationDate']).sum())

print(df[df['Country/Region'] == 'Italy'].groupby(['ObservationDate']).sum())
d1 = datetime.strptime('01/22/2020', "%m/%d/%Y")

d2 = datetime.strptime('03/08/2020', "%m/%d/%Y")

d3 = datetime.strptime('02/26/2020', "%m/%d/%Y")

days_to_subtract_us = abs((d2 - d1).days)

days_to_subtract_italy = abs((d3 - d1).days)

print(days_to_subtract_us)

print(days_to_subtract_italy)
df2 = df[df['Country/Region'].isin(['US', 'Mainland China', 'Italy'])].reset_index(drop=True)

df2 = df2[(df2['Country/Region'] == 'Mainland China') | 

          ((df2['Country/Region'] == 'US') & (df2['ObservationDate'] >= '03/08/2020')) | 

          ((df2['Country/Region'] == 'Italy') & (df2['ObservationDate'] >= '02/26/2020'))].reset_index(drop=True)
def day_number(x):

    if x[0] == 'US':

        return abs((datetime.strptime(x[1], "%m/%d/%Y") - (datetime.strptime('01/22/2020', "%m/%d/%Y"))).days) - days_to_subtract_us

    elif x[0] == 'Italy':

        return abs((datetime.strptime(x[1], "%m/%d/%Y") - (datetime.strptime('01/22/2020', "%m/%d/%Y"))).days) - days_to_subtract_italy

    if x[0] == 'Mainland China':

        return abs((datetime.strptime(x[1], "%m/%d/%Y") - (datetime.strptime('01/22/2020', "%m/%d/%Y"))).days)



df2['DayNumber'] = df2[['Country/Region', 'ObservationDate']].apply(day_number, axis=1)
countries = ['US', 'Mainland China', 'Italy']

for country in countries:

    tmp = df2[df2['Country/Region'] == country].groupby(['DayNumber']).sum()

    plt.plot(tmp.Confirmed, label=country)

    plt.xlabel("Days Since 500 Confirmed Cases")

    plt.ylabel("Confirmed Cases")

    plt.title("Confirmed Case Trends After First 500 Cases \nin the US, Italy, and China")

    plt.legend(loc = 'upper left')