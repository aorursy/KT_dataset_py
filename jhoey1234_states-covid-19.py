import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import calendar

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
months_fmt = mdates.DateFormatter('%B')
us_states = pd.read_csv('../input/usstatescovid/us-states.csv', parse_dates=True) #dataset of cases in the united states
us_regions = pd.read_csv('../input/usa-states-to-region/states.csv') #metadata of states 
#us_states.head()

#convert date column to datetime series
us_states['date'] = pd.to_datetime(us_states['date'])
us_states['new_cases'] = us_states.groupby('state')['cases'].diff().fillna(us_states['cases']) #create new column for new cases per day by state 
us_states['new_deaths'] = us_states.groupby('state')['deaths'].diff().fillna(us_states['deaths']) #create new column for new cases per day by state 
new_cases_table = us_states.pivot_table(values='new_cases', index='date', columns='state') #create pivot table of new cases by state
#merge region informaiton into data frame with confirmed cases count
cases_with_region = pd.merge(us_states, us_regions, left_on='state', right_on='State')
#cases_with_region.head()
#rolling 7 day average of new cases
cases_with_region['7D-RollMean-NC'] = cases_with_region.groupby('state')['new_cases'].transform(lambda x: x.rolling(7, 7).mean()).fillna(0)
cases_with_region['7D-RollMean-ND'] = cases_with_region.groupby('state')['new_deaths'].transform(lambda x: x.rolling(7, 7).mean()).fillna(0)
avg_new_cases_table = cases_with_region.pivot_table(values='7D-RollMean-NC', index='date', columns='state') #create pivot table of new cases by state
#cases_with_region
total_cases = cases_with_region.groupby('date').sum()['cases']
total_new_cases = total_cases.diff().fillna(1)
total_7D_new_cases_avg = total_new_cases.transform(lambda x: x.rolling(7,7).mean()).fillna(0)
sns.relplot(kind='line', data=total_7D_new_cases_avg)
#print(total_7D_new_cases_avg)
#total sum of cases by state
cum_state_cases = us_states.groupby('state')['cases'].max()
#bar chart of cumaltive cases per states
cum_state_cases.plot(kind='bar', fontsize=5, rot=90)
plt.ylabel('Cases');
plt.title ('Total Confirmed Cases by State');
#plt.show()
# plt.clf()
new_cases_sort = avg_new_cases_table.loc[avg_new_cases_table.index[-1]].sort_values()
new_cases_sort = list(new_cases_sort.index)
print(new_cases_sort)
#new_cases_table.plot.area(legend=None)
fig, ax = plt.subplots()

ax.stackplot(avg_new_cases_table.index, avg_new_cases_table[new_cases_sort].T)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
plt.show()
plt.clf()
#bar chart of total cases by region 
cases_by_region = cases_with_region.groupby(['Region', 'date']).sum()[['cases']]
total_cases_by_region = cases_by_region.groupby(['Region']).max()
total_cases_by_region.plot(kind='bar')
plt.show()
plt.clf()
cases_by_region_pivot = cases_by_region.pivot_table(values='cases', index='date', columns='Region').fillna(0) #pivot table of cases for each region
cases_by_region_pivot.plot()
plt.show()
plt.clf()
cases_states_region = cases_with_region.pivot_table(values='cases', index='date', columns=['Region','state']).fillna(0) #pivot table of cases with region and state
cases_states_region_march = cases_states_region.loc['2020-03-01'::] #all entries after March 1
#cases_states_region_march.head()
fig, ax = plt.subplots(2,2 , figsize=(15,15))
fig.subplots_adjust(wspace = 0.5, hspace=0.25)

ax[0,0].plot(cases_states_region_march['Midwest'])
ax[0,0].set_title('Midwest')
ax[0,0].xaxis.set_major_locator(months)
ax[0,0].xaxis.set_major_formatter(months_fmt)

ax[0,1].plot(cases_states_region_march['West'])
ax[0,1].set_title('West')
ax[0,1].xaxis.set_major_locator(months)
ax[0,1].xaxis.set_major_formatter(months_fmt)

ax[1,0].plot(cases_states_region_march['Northeast'])
ax[1,0].set_title('Northeast')
ax[1,0].xaxis.set_major_locator(months)
ax[1,0].xaxis.set_major_formatter(months_fmt)

ax[1,1].plot(cases_states_region_march['South'])
ax[1,1].set_title('South')
ax[1,1].xaxis.set_major_locator(months)
ax[1,1].xaxis.set_major_formatter(months_fmt)



plt.show()
plt.clf()
#percent change over 7 days in cases
#cases_with_region['7 Day Percent Change'] = cases_with_region.groupby('state')['cases'].pct_change(7).fillna(0)*100
#cases_with_region
#group cumaltive cases by month
total_each_month = cases_with_region.groupby(cases_with_region['date'].dt.month).sum()[['new_cases']]
total_each_month.reset_index(inplace=True)
total_each_month['date'] = total_each_month['date'].apply(lambda x: calendar.month_name[x])
#bar chart of new cases every month

sns.catplot(kind='bar', data=total_each_month, x='date', y='new_cases')
plt.show()
plt.clf
#create cumalitive numbers by date
totals = cases_with_region.groupby('date').sum()
totals.reset_index(inplace=True)
#create visualization of new cases and new deaths over time
fig, axes = plt.subplots(1, 2, figsize=(15,10))
fig.subplots_adjust(wspace = 0.25, hspace=0.15)
sns.lineplot(data=totals, x='date', y='new_cases', alpha=0.25, ax=axes[0])
sns.lineplot(data=totals, x='date', y='new_deaths', alpha=0.25, ax=axes[1])
sns.lineplot(data=totals, x='date', y='7D-RollMean-NC', ax=axes[0])
sns.lineplot(data=totals, x='date', y='7D-RollMean-ND', ax=axes[1])


plt.show()
plt.clf()
days = us_states.groupby('date').sum()
days.head()
days.loc['2020-02-22']
ny_avg_new_cases = avg_new_cases_table.loc['2020-07-01':]['New York'].fillna(0)
ny_new_cases = new_cases_table.loc['2020-07-01':]['New York'].fillna(0)
ny_new_cases.plot(alpha=0.2)
ny_avg_new_cases.plot()
plt.show()
plt.clf()
nj_avg_new_cases = avg_new_cases_table.loc['2020-07-01':]['New Jersey'].fillna(0)
nj_new_cases = new_cases_table.loc['2020-07-01':]['New Jersey'].fillna(0)
nj_new_cases.plot(alpha=0.2)
nj_avg_new_cases.plot()
plt.show()
plt.clf()
