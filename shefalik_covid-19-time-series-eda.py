# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
df
df.info()
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=False)
df.info()
df1 = df.groupby(df["Country/Region"])
df1.head(10)
df1.groups.keys()
df1.size().sort_values()   #To count rows in each group - group.size()
df1.get_group('China')
by_month = df.groupby(pd.Grouper(key='Date',freq='M')).size()
by_month
plt.plot(by_month)
latest = df.drop(['Lat','Long'],axis=1)
daily_latest = latest.groupby(pd.Grouper(key='Date',freq = 'D')).sum()
daily_latest
# fatality_rate = 
conf = daily_latest['Confirmed'][-1]
death = daily_latest['Deaths'][-1]
rec = daily_latest['Recovered'][-1]
fatality_rate = (death/conf)*100
survival_rate = (rec/conf)*100
print("Fatality rate: ", fatality_rate)
print("Survival rate: ", survival_rate)
labels = 'Deaths', 'Recovered', 'Current Cases'
sizes = [fatality_rate, survival_rate, 100-(fatality_rate+survival_rate)]
explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
plt.plot(daily_latest)
weekly_data = latest.groupby(pd.Grouper(key='Date',freq = 'W')).sum()
weekly_data
weekly_spike = weekly_data.diff(axis=0)  #to get the growth in number of cases between different weeks
weekly_spike
weekly_spike.dropna()  #to drop the first row that contains nan value
weekly_spike.plot.bar(rot=0,subplots=True)
plt.plot(weekly_spike)
df2 = df.groupby(df["Date"])
df2.head()
df2.groups.keys()
total_cases = df2.get_group(df['Date'].iloc[-1])
total_cases
total_cases = total_cases.groupby(total_cases['Country/Region']).sum()  #to sum up the total cases in the country, by different provinces in the county as well
total_cases
total_cases = total_cases.sort_values('Confirmed',ascending=False)      #sort such that highest confirmed cases are shown first
total_cases.head(10)
top_10 = total_cases.head(10).drop(['Lat','Long'],axis=1)
top_10.plot.bar()
