import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mpl_dates

import matplotlib.ticker as ticker
df = pd.read_csv('../input/covid19-dataset-jansept-2020-us/national-history.csv', index_col=0)
df = df[['death', 'deathIncrease', 'negative', 'negativeIncrease', 'positive', 'positiveIncrease', 'recovered', 'totalTestResults', 'totalTestResultsIncrease']]

df.rename(columns={'deathIncrease': 'new_deaths', 'positiveIncrease': 'new_positve', 'totalTestResultsIncrease': 'new_tests'}, 

         inplace=True)

df.fillna(0, inplace=True)

df.index = pd.to_datetime(df.index)

df.sort_index(inplace=True)
dates = df.index

daily_cases = df['new_positve']



plt.style.use('seaborn')



plt.figure(figsize=(20, 12), dpi=300)

plt.title('Daily Covid-19 Positve Cases in the US (Jan-Sept, 2020)', fontsize=20)



date_format = mpl_dates.DateFormatter('%d %b')

plt.gca().xaxis.set_major_formatter(date_format)



plt.xticks(fontsize=13)

plt.yticks(fontsize=13)



plt.plot(dates, daily_cases, color='#00008B', marker='.');
daily_tests = df['new_tests']



plt.figure(figsize=(20, 12), dpi=300)

plt.title('Covid-19 Daily Tests & Postive Cases in the US (Jan-Sept, 2020)', fontsize=20)



date_format = mpl_dates.DateFormatter('%d %b')

plt.gca().xaxis.set_major_formatter(date_format)

plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())





plt.plot(dates, daily_tests, color='#FF8C00', marker='.', label='Daily Tests');

plt.plot(dates, daily_cases, color='#008080', marker='.', label='Daily Positive Cases');



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)





plt.legend(fontsize=18);

plt.show();
daily_deaths = df['new_deaths']



plt.figure(figsize=(20, 12), dpi=300)

plt.title('Covid-19 Daily Deaths in the US (Jan-Sept, 2020)', fontsize=20)



date_format = mpl_dates.DateFormatter('%d %b')

plt.gca().xaxis.set_major_formatter(date_format)





plt.plot(dates, daily_deaths, color='#800000', marker='.');



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.show();
plt.figure(figsize=(20, 12), dpi=300)

plt.title('Covid-19 Daily Cases & Postive Deaths in the US (Jan-Sept, 2020)', fontsize=20)



date_format = mpl_dates.DateFormatter('%d %b')

plt.gca().xaxis.set_major_formatter(date_format)

plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())





plt.plot(dates, daily_cases, color='#008080', marker='.', label='Daily Positive Cases');

plt.plot(dates, daily_deaths, color='#800000', marker='.', label='Daily Deaths');



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)





plt.legend(fontsize=18);

plt.show();
states_df = pd.read_csv('../input/covid19-dataset-jansept-2020-us/all-states-history.csv', index_col=0)
states_df = states_df[['state', 'death', 'deathIncrease', 'positive', 'positiveIncrease', 'totalTestResults',

       'totalTestResultsIncrease' ]]

states_df.sort_index(inplace=True)

states_df.index = pd.to_datetime(states_df.index)

states_df.fillna(0, inplace=True)
states_grp = states_df.groupby('state')

total_case_counts = states_grp['positiveIncrease'].sum()

sorted_states = total_case_counts.sort_values(ascending=False)

sorted_states = sorted_states.to_dict()
states = [ s for s in list(sorted_states) ]

total_cases = [ sorted_states[i] for i in list(sorted_states) ]



plt.figure(figsize=(20,12), dpi=300)

plt.title('Covid-19 Total Positive Cases in Every States of USA (Jan-Sept, 2020)', fontsize=20)



plt.barh(states, total_cases, color='#FF6347');

plt.gca().invert_yaxis()
fig, ax = plt.subplots(10, 5, sharex=True)

fig.suptitle('Daily Cases in US States', fontsize=25)



fig.set_figheight(30)

fig.set_figwidth(30)



date_format = mpl_dates.DateFormatter('%b')





i = 0

j = 0



for name, dfs in states_grp:

    

    x = dfs.index

    y = dfs['positiveIncrease']

    

    if i < 10: 

        ax[i,j].plot(x, abs(y), color='#008080');

        ax[i,j].set_title(f'{name}', fontsize=17);

        

        ax[i,j].xaxis.set_major_formatter(date_format);

        j += 1

        if j > 4:

            i +=1

            j = 0





plt.show();