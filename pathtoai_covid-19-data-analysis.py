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
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

train['Date'] = pd.to_datetime(train['Date'])

display(train.head(5))

display(train.describe())

train.dtypes

from datetime import date as dt

train['week'] = train['Date'].dt.week
print("Number of Country/Region: ", train['Country/Region'].nunique())

print("Dates go from day : ", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed:  ", train[train['Province/State'].isna()==False]['Country/Region'].unique())
grp_count = train.groupby('Country/Region')[['ConfirmedCases','Fatalities']].max()

grp_count['Death_per_Case%'] = (grp_count['Fatalities'] / grp_count['ConfirmedCases'] )*100

grp_count.sort_values('ConfirmedCases', ascending=False).head(20)

train[train['Country/Region'] == 'India']['ConfirmedCases'].max()
import matplotlib.pyplot as plt

confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_noChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_noChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_india = train[train['Country/Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_india = train[train['Country/Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_india = confirmed_total_date_india.join(fatalities_total_date_india)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_india.plot(ax=ax1)

ax1.set_title("Global confirmed cases in India", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_india.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases in India", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_china = train[train['Country/Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_china = train[train['Country/Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_china = confirmed_total_date_china.join(fatalities_total_date_china)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_china.plot(ax=ax1)

ax1.set_title("Global confirmed cases in India", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_china.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases in India", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
import matplotlib.pyplot as plt

confirmed_total_date = train.groupby(['week']).agg({'ConfirmedCases':['max']})

fatalities_total_date = train.groupby(['week']).agg({'Fatalities':['max']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Week", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Week", size=13)
confirmed_total_date_india = train[train['Country/Region']=='India'].groupby(['week']).agg({'ConfirmedCases':['max']})

fatalities_total_date_india = train[train['Country/Region']=='India'].groupby(['week']).agg({'Fatalities':['max']})

total_date_india = confirmed_total_date_india.join(fatalities_total_date_india)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_india.plot(ax=ax1)

ax1.set_title("Global confirmed cases in India", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Week", size=13)

fatalities_total_date_india.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases in India", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Week", size=13)
confirmed_total_date_china = train[train['Country/Region']=='China'].groupby(['week']).agg({'ConfirmedCases':['max']})

fatalities_total_date_china = train[train['Country/Region']=='China'].groupby(['week']).agg({'Fatalities':['max']})

total_date_china = confirmed_total_date_china.join(fatalities_total_date_china)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_china.plot(ax=ax1)

ax1.set_title("Global confirmed cases in India", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Week", size=13)

fatalities_total_date_china.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases in India", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Week", size=13)