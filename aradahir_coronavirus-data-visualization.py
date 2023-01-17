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

        #print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
group_table = df.groupby(['Country/Region']).sum()

table = group_table.reset_index()

table['In hospital cases'] = table.Confirmed - (table.Recovered+table.Deaths)

table['Percent Infection'] = table.Confirmed*100/sum(table.Confirmed)

table['Percent Recovery'] = table.Recovered*100/table.Confirmed

table['Percent Mortality'] = table.Deaths*100/table.Confirmed
import seaborn as sns

import matplotlib.pyplot as plt
Top_rank_infected_country = table.sort_values(by='Confirmed', ascending=False)

Top_rank_infected_country.head(10)
Top_rank_case_country = table.sort_values(by='In hospital cases', ascending=False)

Top_rank_case_country.head(10)
figure_size = (20, 4)

fig, ax = plt.subplots(figsize=figure_size)

bar = sns.barplot(x="Country/Region", y="Percent Recovery", data=table.sort_values(by='Percent Recovery', ascending=False))

bar.set_xticklabels(bar.get_xticklabels(),rotation=90)

bar.set_title('Percent Recovery by Country')
figure_size = (20, 4)

fig, ax = plt.subplots(figsize=figure_size)

bar = sns.barplot(x="Country/Region", y="Percent Mortality", data=table.sort_values(by='Percent Mortality', ascending=False))

bar.set_xticklabels(bar.get_xticklabels(),rotation=90)

bar.set_title('Percent Mortality by Country')
Mainland_China = df[df['Country/Region'] == 'Mainland China'].groupby(['Province/State']).sum()

Mainland_China['percent infect'] = Mainland_China['Confirmed'] *100/ sum(Mainland_China['Confirmed'] )

Mainland_China['Percent Recovery'] = Mainland_China.Recovered*100/Mainland_China.Confirmed

Mainland_China['Percent Mortality'] = Mainland_China.Deaths*100/Mainland_China.Confirmed



Top_rank_infected_Province = Mainland_China.sort_values(by='Confirmed', ascending=False)

Top_rank_infected_Province.head(10)


time = df.groupby(['Date']).sum().reset_index()

time['In hospital cases'] = time.Confirmed - (time.Recovered+time.Deaths)

time_for_plot = time.drop(columns = ['Lat','Long','Confirmed'])

time_for_plot

figure_size = (20, 10)

fig, ax = plt.subplots(figsize=figure_size)

plt.title('Coronavirus cases per day')

stacked_bar = time_for_plot.plot(kind='bar', stacked=True, ax=ax)

plt.xticks(time_for_plot.index, time_for_plot['Date'])

plt.xlabel('Date of Record')

plt.ylabel('Number of cases')
