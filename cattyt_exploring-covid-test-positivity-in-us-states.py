from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.figure as fg # figure control

import matplotlib.pyplot as plt # plotting

import matplotlib.ticker as plticker  #ticker control

import numpy as np

import os

import pandas as pd



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/covid19testing/tested_worldwide.csv', delimiter=',') 

df1.dataframeName = 'tested_worldwide.csv'

us = df1[df1.Country_Region=='United States']

all_states = np.sort(us.Province_State.unique())

raw_states = np.delete(all_states,[2,3,13,38,43,51])

print("States included:")

states_list = ''

for i in range(0,raw_states.size):

    if i == raw_states.size-1:

        states_list += raw_states[i]

    else:

        states_list += raw_states[i]+', '

print(states_list)
a = df1[df1.Province_State=='Alabama']

a.tail()
fig, axs = plt.subplots(9,6,gridspec_kw={'hspace': 0.4, 'wspace': 0.3},figsize=(20, 20))



for i, ax in enumerate(fig.axes):

    if i>50:

        ax.set_visible(False)

    else:

        state = df1[df1.Province_State==raw_states[i]]

        short_dates = []

        for j in range(0,state.Date.values.size):

            short_dates.append(state.Date.values[j].replace('2020-',''))

        ax.plot(short_dates, state.daily_tested,'-')

        ax.plot(short_dates, state.daily_positive,'-')

        ax.set_title(raw_states[i])

        if i!=0:

            ax.get_xaxis().set_visible(False)

        else:

            #for tick in ax.get_xticklabels():

            #    tick.set_rotation(90)

            loc = plticker.MultipleLocator(base=20.0)

            ax.xaxis.set_major_locator(loc)



fig.suptitle('Daily Total Tests and Positive Tests By State (Raw)')



plt.show()
# Calculate test-positive rates per state

us=us.assign(daily_rate=(us.daily_positive/us.daily_tested)*100)

us.daily_rate.replace(us[us.daily_rate<0].daily_rate, 0, inplace=True)

us['daily_rate'].clip(lower=0)

us.daily_rate.fillna(0, inplace=True)
fig, axs = plt.subplots(9,6,sharey=True,gridspec_kw={'hspace': 0.4, 'wspace': 0.4},figsize=(20, 25))



for i, ax in enumerate(fig.axes):

    if i>50:

        ax.set_visible(False)

    else:

        state = us[us.Province_State==raw_states[i]]

        short_dates = []

        for j in range(0,state.Date.values.size):

            short_dates.append(state.Date.values[j].replace('2020-',''))

        daily_tested_rolling = state.daily_tested.rolling(window=7).mean()

        daily_rate_rolling = state.daily_rate.rolling(window=7).mean()

        ax.plot(short_dates, daily_rate_rolling,'-', color='tab:orange')

        

        ax2 = ax.twinx()

        ax2.plot(short_dates, daily_tested_rolling,'-')

        ax2.yaxis.set_major_locator(plticker.MaxNLocator(nbins=5))

        ax.set_title(raw_states[i])

        #ax.get_xaxis().set_visible(False)

        if i!=0:

            ax.get_xaxis().set_visible(False)

        else:

            #for tick in ax.get_xticklabels():

            #    tick.set_rotation(90)

            loc = plticker.MultipleLocator(base=20.0)

            ax.xaxis.set_major_locator(loc)



fig.suptitle('Daily Test-Positivity Rate (orange) and Total Tests (blue) By State (Rolling)')



plt.show()
# Add population data for per capita testing

popdata = pd.read_csv('/kaggle/input/nstest2019alldata/nst-est2019-alldata.csv', delimiter=',')
# Calculate rolling test-positive rates per state

us=us.assign(daily_rate=(us.daily_positive/us.daily_tested)*100)

us.daily_rate.replace(us[us.daily_rate<0].daily_rate, 0, inplace=True)

us['daily_rate'].clip(lower=0)

us.daily_rate.fillna(0, inplace=True)
fig, axs = plt.subplots(9,6,gridspec_kw={'hspace': 0.4, 'wspace': 0.3},figsize=(20, 20),sharey=True)

#fig, axs = plt.subplots(7,8,gridspec_kw={'hspace': 0.4, 'wspace': 0.3},figsize=(30, 20),sharey=True)

#fig, axs = plt.subplots(1,6,gridspec_kw={'hspace': 0.4, 'wspace': 0.3},figsize=(20, 5),sharey=True)

all_tests_per_capita = pd.DataFrame()



for i, ax in enumerate(fig.axes):

    # Hide blank plots since 51 doesn't divide evenly into full rows

    if i>50:

        ax.set_visible(False)

    else:

        state_name = raw_states[i]

        state = us[us.Province_State==state_name]

        #short_dates = []

        #for j in range(0,state.Date.values.size):

        #    short_dates.append(state.Date.values[j].replace('2020-',''))

        short_dates = state.Date.str.replace('2020-','')

        daily_tested_rolling = state.daily_tested.rolling(window=7).mean()

        pop = popdata[popdata.NAME==state_name].POPESTIMATE2019.values[0]

        tests_per_capita = pd.Series(daily_tested_rolling.divide(pop)*100000,name=state_name)

        tests_per_capita.index=state.Date # Set index to dates; does not work in initialization

        all_tests_per_capita = all_tests_per_capita.append(tests_per_capita)

        

        ax.plot(short_dates, tests_per_capita,'-')

        ax.axhline(y=152,ls='--')

        ax.set_title(state_name)

        if i==0:

            loc = plticker.MultipleLocator(base=20.0)

            ax.xaxis.set_major_locator(loc)

        else:

            ax.get_xaxis().set_visible(False)



        latest = tests_per_capita[-1:].values[0]

        ax.annotate(int(latest), xy=(state.Date.max(),latest), xytext=(-20,10), textcoords='offset points')



fig.suptitle('Daily Number of Tests Per Capita (100,000) By State')



plt.show()
#Calculate mean daily rates and total tested for all states, for the last seven days

# Get only last seven days

last_week=us['Date'].unique()[-7:]

d = us[['Province_State','daily_rate','daily_tested','death','Date']]

last_week_data = d[d['Date'].isin(last_week)]



# Remove Maine because of low reporting

trimmed_states = np.delete(raw_states,np.where(raw_states=='Maine'))



# Final mean arrays to plot

mean_daily_rate = last_week_data[['Province_State','daily_rate']].groupby('Province_State').mean().loc[trimmed_states]

mean_daily_tested = last_week_data[['Province_State','daily_tested']].groupby('Province_State').mean().loc[trimmed_states]



# Calculate total number of deaths from the last week

d = last_week_data.sort_values(by=['Province_State', 'death', 'Date'])

d['last_week_deaths'] = d.groupby('Province_State')['death'].diff(periods=6)

total_deaths = d[d['Date'].isin([last_week[-1]])][['Province_State','last_week_deaths']].set_index('Province_State').loc[trimmed_states]
# Mean tests per capita last seven days

mean_daily_tpc = all_tests_per_capita[all_tests_per_capita.columns.sort_values()[-7:]].mean(axis=1)

mean_daily_tpc = mean_daily_tpc[mean_daily_tpc.index!='Maine']
fig, ax = plt.subplots(figsize=(30, 12))

#ax.scatter(mean_daily_rate, mean_daily_tested, s=last_week_tests_per_capita)#, c=close, s=volume, alpha=0.5)

ax.scatter(mean_daily_rate, mean_daily_tpc, s=mean_daily_tested/10, c=total_deaths.values, alpha=0.5)

#ax.scatter(mean_daily_rate, mean_daily_tpc, s=mean_daily_tested/10, alpha=0.5)



for state in mean_daily_rate.index:

    ax.annotate(state,xy=(mean_daily_rate.loc[state],mean_daily_tpc.loc[state]))



ax.set_xlabel('Average Test-Positivity Rate')

ax.set_ylabel('Average Tests Per Capita (100,000)')

ax.set_title('Average Daily Test-Positivity Rate vs. Tests Per Capita By State, Colored by Total Deaths, Over The Past Week')



ax.grid(True)

plot = ax.pcolor(total_deaths.values,visible=False)

fig.colorbar(plot)

plt.show()