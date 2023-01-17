# importing all the essentials library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Reading and Assigning the DataFrame to df variable

df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
# checking the latest 7 days covid data

df.tail(7)
# Checking for any NA value and removing, sometimes this data contains NA value

na = df.isna().sum()

for index in na.index:

    if na.loc[index] > 0:

        df = df.dropna().reset_index(drop=True)
# Adding the Year in Date Columns & Converting it to Pandas Datetime Format

df['Date'] = df['Date'] + '2020' # Please Run this once else it will cause error by adding '2020' multiple times

df['Date'] = pd.to_datetime(df['Date'])
# Adding Two Columns for Mortality & Recovery Rate

df.loc[:, 'Mortality'] = ((df.loc[:, 'Total Deceased'] / df.loc[:, 'Total Confirmed']) * 100)

df.loc[:, 'Recovery'] = ((df.loc[:, 'Total Recovered'] / df.loc[:, 'Total Confirmed']) * 100)
# Assigning data to variables for easy plotting

date = df.loc[:, 'Date']

recovery = df.loc[:, 'Recovery'] / 10 # Normalised to fit in single chart

mortality = df.loc[:, 'Mortality']

confirmed = df.loc[:, 'Daily Confirmed'] / 10000 # Normalised to fit in single chart
plt.xkcd()

fig, ax = plt.subplots(figsize=(20, 10))



ax.plot(date, mortality, 'r.-', label='Mortality Rate')

ax.plot(date, recovery, 'g.-', label='Recovery Rate (10x)')

ax.plot(date, confirmed, 'b.-', label='Daily Cases (10,000x)')



ax.set_ylabel('Trend', fontsize=20)

ax.set_title('Covid Trend in India', fontsize=40)

ax.set_xlabel('Months', fontsize=20)

ax.set_xticklabels(['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], fontsize=40)

ax.set_yticklabels(np.arange(0, 12, 2), fontsize=40)

ax.set_yticks(np.arange(0, 12, 2))



ax.legend(fontsize=25)

ax.grid(True, linewidth=1)



# plt.savefig('test1.png') # For Saving The Picture

plt.show()