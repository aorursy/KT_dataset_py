# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import matplotlib.dates as mdates





plt.style.use('ggplot')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')

df.head()
df.info(verbose=True)
df.Country.unique()
# focus in hong kong 

hk = df[df.Country == 'Hong Kong SAR, China']

hk.head()
# When was the first case in hong kong 

print(f'First Case in Hong Kong on the {hk.Date.min()}')

# When was the last case in hong kong 

print(f'Last Case in Hong Kong on the {hk.Date.max()}')
total_cases = hk.iloc[-1, -3]

print(f'Total number of confirmed : {total_cases}')


# Since the number of deaths were accumulated, we can just look at the last record to find out the total death.

total_death = hk.iloc[-1, -2]

print(f'Total number of death : {total_death}')
print(f'SARs Death Rate : {total_death / total_cases * 100 :.2f}%')
total_recover = hk.iloc[-1, -1]

print(f'Total number of recovered: {total_recover}')

print(f'SARs Recover Rate in Hong Kong : {total_recover / total_cases * 100 :.2f}%')
hk.Date = pd.to_datetime(hk.Date)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 15), sharex=True)

hk.set_index('Date')['Cumulative number of case(s)'].plot.line(ax=ax1)

ax1.set_xlabel('Date')

ax1.set_ylabel('Cumulated Number of case(s)')

ax1.set_title('How the absolute number of confirmed cases increase over the period')



# Growth rate

ax2.plot(hk.Date, np.log2(hk['Cumulative number of case(s)']))

ax2.set_xlabel('Date')

ax2.set_ylabel('Cumulated Number of case(s)')

ax2.set_title('How log2 growth rate over the period')

# set monthly locator

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

ax2.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

# set formatter

ax2.xaxis.set_major_formatter(plt.NullFormatter())

ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%d-%m-%Y'))

ax2.set_xlim('17-03-2003', '15-07-2003')



fig.autofmt_xdate()

plt.show()