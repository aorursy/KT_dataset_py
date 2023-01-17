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
data = pd.read_excel('/kaggle/input/covid19india/covidDatabaseIndia.xls', sheet_name = 'Cases_Time_Series')
data.head()
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
ax = sns.lineplot(x = 'Date', y = 'Daily Confirmed', data = data)
ax.set_title('Daily Confirmed Cases')
ax = sns.lineplot(x = 'Date', y = 'Total Confirmed', data = data)
ax.set_title('Total Confirmed Cases')
ax = sns.barplot(x = 'Date', y = 'Daily Recovered', data = data)
ax.set_title('Daily Recovered Cases')
ax = sns.lineplot(x = 'Date', y = 'Total Recovered', data = data)
ax.set_title('Total Recovered Cases')
ax = sns.barplot(x = 'Date', y = 'Daily Deceased', data = data)
ax.set_title('Daily Deceased Cases')
ax = sns.lineplot(x = 'Date', y = 'Total Deceased', data = data)
ax.set_title('Total Deceased Cases')
sns.lineplot(x = 'Date', y = 'Total Confirmed', data = data)
sns.lineplot(x = 'Date', y = 'Total Deceased', data = data)
sns.lineplot(x = 'Date', y = 'Total Recovered', data = data)


state = pd.read_excel('/kaggle/input/covid19india/covidDatabaseIndia.xls', sheet_name = 'Statewise')
state = state.drop(state.index[0])
state.head()
fig = sns.barplot(x = 'State', y = 'Confirmed', data = state)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig.set_title('Statewise Total Cases')
fig = sns.barplot(x = 'State', y = 'Recovered', data = state)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig.set_title('Statewise Recovered Cases')
fig = sns.barplot(x = 'State', y = 'Deaths', data = state)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig.set_title('Statewise Deaths')
f, ax = plt.subplots(figsize=(6, 10))
sns.set_color_codes("pastel")
sns.barplot(x = 'State', y = 'Confirmed', data = state, color = 'b')
sns.set_color_codes("muted")
sns.barplot(x = 'State', y = 'Recovered', data = state, color = 'b')
sns.set_color_codes("muted")
sns.barplot(x = 'State', y = 'Deaths', data = state, color = 'k')
ax.set_xticklabels(fig.get_xticklabels(), rotation=90)
sns.despine(left=True, bottom=True)
icmrc = pd.read_excel('/kaggle/input/covid19india/covidDatabaseIndia.xls', sheet_name = 'Tested_Numbers_ICMR_Data')
icmrc.head()
icmr = pd.DataFrame(icmrc['Update Time Stamp'])
icmr['Total Samples Tested'] = icmrc['Total Samples Tested']
icmr.head()
fig = sns.lineplot(x = 'Update Time Stamp', y = 'Total Samples Tested', data = icmr)
fig.set_xticklabels(labels = fig.get_xticklabels(), rotation=90)
fig = sns.lineplot(x = 'Update Time Stamp', y = 'Total Samples Tested', data = icmr, color = 'b')
sns.lineplot(x = 'Date', y = 'Total Confirmed', data = data[43:], color = 'r')
fig.set_xticklabels(labels = fig.get_xticklabels(), rotation=90)
plt.legend(labels=['Total Tests', 'Total Cases'])
