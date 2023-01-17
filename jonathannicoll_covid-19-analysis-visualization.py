import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
confirmed_global = pd.read_csv('../input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
confirmed_global = confirmed_global.drop(columns=['Lat', 'Long'])
confirmed_global['Total Confirmed'] = confirmed_global.iloc[:,-1]
def missing_values_table(confirmed_global):
    mis_val = confirmed_global.isnull().sum()
    mis_val_pct = 100*mis_val / len(confirmed_global)
    mis_val_table = pd.concat([mis_val, mis_val_pct], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0: 'Missing Values', 1: '% of Total  Values'})
    return mis_val_table_ren_columns

missing_confirmed_global = missing_values_table(confirmed_global)
missing_confirmed_global.sort_values('Missing Values', ascending=False).head()
confirmed_global = confirmed_global.groupby('Country/Region').sum().reset_index()
confirmed_global.head()
deaths_global = pd.read_csv('../input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
deaths_global = deaths_global.drop(columns=['Lat', 'Long'])
deaths_global['Total Deaths'] = deaths_global.iloc[:,-1]
missing_confirmed_deaths = missing_values_table(deaths_global)
missing_confirmed_deaths.sort_values('Missing Values', ascending=False).head()
deaths_global = deaths_global.groupby('Country/Region').sum().reset_index()
deaths_global.head()
recovered_global = pd.read_csv('../input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
recovered_global = recovered_global.drop(columns=['Lat', 'Long'])
recovered_global['Total Recovered'] = recovered_global.iloc[:,-1]
missing_recovered_global = missing_values_table(recovered_global)
missing_recovered_global.sort_values('Missing Values', ascending=False).head()
recovered_global = recovered_global.groupby('Country/Region').sum().reset_index()
recovered_global.head()
latest_confirmed_global = confirmed_global[['Country/Region', 'Total Confirmed']]
latest_confirmed_deaths = deaths_global[['Country/Region', 'Total Deaths']]
latest_confirmed_recovered = recovered_global[['Country/Region', 'Total Recovered']]

latest_global_stats = latest_confirmed_global.merge(latest_confirmed_deaths, on=['Country/Region'], how='left').merge(latest_confirmed_recovered, on=['Country/Region'], how='left')

latest_global_stats['Active Cases'] = latest_global_stats['Total Confirmed'] - latest_global_stats['Total Deaths'] - latest_global_stats['Total Recovered']
latest_global_stats['Mortality Rate (%)'] = (latest_global_stats['Total Deaths'] / latest_global_stats['Total Confirmed']*100).round(2)
latest_global_stats.to_csv('latest_global_stats.csv')
latest_global_stats.head()
missing_latest_global = missing_values_table(latest_global_stats)
missing_latest_global.sort_values('Missing Values', ascending=False).head()
worldwide_stats = latest_global_stats[['Total Confirmed', 'Total Deaths', 'Total Recovered', 'Active Cases']].sum()
worldwide_stats = worldwide_stats.to_frame().rename(columns={0: 'Number of Cases'})
worldwide_stats
total_confirmed = worldwide_stats.iloc[0,0]
total_deaths = worldwide_stats.iloc[1,0]
total_recovered = worldwide_stats.iloc[2,0]
total_active = worldwide_stats.iloc[3,0]
labels = ['Deaths', 'Recovered', 'Active']
colors = ['red', 'green', 'orange']
values = [total_deaths, total_recovered, total_active]
fig, ax = plt.subplots(figsize=(8,6))
ax.pie(values, explode=(0.4, 0, 0), autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14}, colors=colors)
ax.axis('equal')
ax.set_title('Worldwide COVID-19 Cases', size=16)
ax.legend(labels, title='Cases', loc='upper right')
plt.show()
total_global_cases = confirmed_global.sum(axis=0).to_frame()
total_global_cases = total_global_cases.iloc[2:-1].rename(columns={0: 'Total Confirmed Cases'})
days = range(0,len(total_global_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(total_global_cases['Total Confirmed Cases'], linewidth=2)
plt.title('Evolution of Total Confirmed Cases (Worldwide)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('Confirmed Cases')
plt.xticks(days, days)
plt.show()
total_global_deaths = deaths_global.sum(axis=0).to_frame()
total_global_deaths = total_global_deaths.iloc[2:-1].rename(columns={0: 'Total Confirmed Deaths'})
days = range(0,len(total_global_deaths),10)

ax = plt.figure(figsize=(16,6))
plt.plot(total_global_deaths['Total Confirmed Deaths'], linewidth=2)
plt.title('Evolution of Total Confirmed Deaths (Worldwide)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('Confirmed Deaths')
plt.xticks(days, days)
plt.show()
total_global_recoveries = recovered_global.sum(axis=0).to_frame()
total_global_recoveries = total_global_recoveries.iloc[2:-1].rename(columns={0: 'Total Confirmed Recoveries'})
days = range(0,len(total_global_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(total_global_recoveries['Total Confirmed Recoveries'], linewidth=2)
plt.title('Evolution of Total Confirmed Recoveries (Worldwide)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('Confirmed Recoveries')
plt.xticks(days, days)
plt.show()
top_10_confirmed = latest_global_stats.sort_values('Total Confirmed', ascending=False)[:10][['Country/Region', 'Total Confirmed']].reset_index(drop=True)
top_10_confirmed
labels = top_10_confirmed['Country/Region']
confirmed = top_10_confirmed['Total Confirmed']
ax = plt.subplot()
plt.barh(range(len(labels)), confirmed, color='red', edgecolor='black')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.xlabel('Confirmed Cases')
plt.title('Top 10 Countries (Confirmed Cases)')
plt.show()
top_10_deaths = latest_global_stats.sort_values('Total Deaths', ascending=False)[:10][['Country/Region', 'Total Deaths']].reset_index(drop=True)
top_10_deaths
labels = top_10_deaths['Country/Region']
deaths = top_10_deaths['Total Deaths']
ax = plt.subplot()
plt.barh(range(len(labels)), deaths, color='red', edgecolor='black')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.xlabel('Total Deaths')
plt.title('Top 10 Countries (Total Deaths)')
plt.show()
over_1000_cases = latest_global_stats[latest_global_stats['Total Confirmed'] >= 1000]
highest_deathrate = over_1000_cases.sort_values('Mortality Rate (%)', ascending=False)[:10][['Country/Region', 'Mortality Rate (%)']].reset_index(drop=True)
highest_deathrate
labels = highest_deathrate['Country/Region']
deathrate = highest_deathrate['Mortality Rate (%)']
ax = plt.subplot()
plt.barh(range(len(labels)), deathrate, color='red', edgecolor='black')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.xlabel('Mortality Rate (%)')
plt.title('Top 10 Countries (Highest Death Rate (over 1000 cases))')
plt.show()
lowest_deathrate = over_1000_cases.sort_values('Mortality Rate (%)')[:10][['Country/Region', 'Mortality Rate (%)']].reset_index(drop=True)
lowest_deathrate
labels = lowest_deathrate['Country/Region']
deathrate = lowest_deathrate['Mortality Rate (%)']
ax = plt.subplot()
plt.barh(range(len(labels)), deathrate, color='green', edgecolor='black')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.xlabel('Mortality Rate (%)')
plt.title('Top 10 Countries (Lowest Death Rate (over 1000 cases))')
plt.show()
confirmed_cases_country = confirmed_global.groupby('Country/Region').sum().reset_index()
canada_confirmed_cases = confirmed_cases_country[confirmed_cases_country['Country/Region'] == 'Canada'].reset_index(drop=True)
uk_confirmed_cases = confirmed_cases_country[confirmed_cases_country['Country/Region'] == 'United Kingdom'].reset_index(drop=True)
us_confirmed_cases = confirmed_cases_country[confirmed_cases_country['Country/Region'] == 'US'].reset_index(drop=True)
canada_confirmed = canada_confirmed_cases.iloc[0,1:-1].to_frame()
uk_confirmed = uk_confirmed_cases.iloc[0,1:-1].to_frame()
us_confirmed = us_confirmed_cases.iloc[0,1:-1].to_frame()
days = range(0,len(canada_confirmed),10)
legend = ['Canada', 'UK', 'US']

ax = plt.figure(figsize=(16,6))
plt.plot(canada_confirmed, linewidth=2)
plt.plot(uk_confirmed, linewidth=2)
plt.plot(us_confirmed, linewidth=2)
plt.legend(legend)
plt.title('Comparison of Total Confirmed Cases (Canada, UK and US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('Total Confirmed Cases')
plt.xticks(days, days)
plt.show()
deaths_country = deaths_global.groupby('Country/Region').sum().reset_index()
canada_confirmed_deaths = deaths_country[deaths_country['Country/Region'] == 'Canada'].reset_index(drop=True)
uk_confirmed_deaths = deaths_country[deaths_country['Country/Region'] == 'United Kingdom'].reset_index(drop=True)
us_confirmed_deaths = deaths_country[deaths_country['Country/Region'] == 'US'].reset_index(drop=True)
canada_deaths = canada_confirmed_deaths.iloc[0,1:-1].to_frame()
uk_deaths = uk_confirmed_deaths.iloc[0,1:-1].to_frame()
us_deaths = us_confirmed_deaths.iloc[0,1:-1].to_frame()
days = range(0,len(canada_deaths),10)
legend = ['Canada', 'UK', 'US']

ax = plt.figure(figsize=(16,6))
plt.plot(canada_deaths, linewidth=2)
plt.plot(uk_deaths, linewidth=2)
plt.plot(us_deaths, linewidth=2)
plt.legend(legend)
plt.title('Comparison of Total Deaths (Canada, UK and US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('Total Deaths')
plt.xticks(days, days)
plt.show()
canada_latest_stats = latest_global_stats[latest_global_stats['Country/Region'] == 'Canada']
uk_latest_stats = latest_global_stats[latest_global_stats['Country/Region'] == 'United Kingdom']
us_latest_stats = latest_global_stats[latest_global_stats['Country/Region'] == 'US']

countries = ['Canada', 'United Kingdom', 'US']
summary_stats = latest_global_stats.loc[latest_global_stats['Country/Region'].isin(countries)].reset_index(drop=True)
summary_stats
labels = ['Deaths', 'Recoveries', 'Active']
colors = ['red', 'green', 'orange']
canada = canada_latest_stats[['Total Deaths', 'Total Recovered', 'Active Cases']]
uk = uk_latest_stats[['Total Deaths', 'Total Recovered', 'Active Cases']]
us = us_latest_stats[['Total Deaths', 'Total Recovered', 'Active Cases']]

plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 3, 1)
ax1.pie(canada, explode=(0.2, 0, 0), autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14}, colors=colors)
ax1.axis('equal')
ax1.set_title('Canada COVID-19 Cases', size=16, style='oblique')
ax1.legend(labels, title='Cases')

ax2 = plt.subplot(1, 3, 2)
ax2.pie(uk, explode=(0.2, 0, 0), autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14}, colors=colors)
ax2.axis('equal')
ax2.set_title('UK COVID-19 Cases', size=16, style='oblique')
ax2.legend(labels, title='Cases')

ax3 = plt.subplot(1, 3, 3)
ax3.pie(us, explode=(0.2, 0, 0), autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14}, colors=colors)
ax3.axis('equal')
ax3.set_title('US COVID-19 Cases', size=16, style='oblique')
ax3.legend(labels, title='Cases')

plt.tight_layout()
plt.show()
new_global_cases = confirmed_global.sum(axis=0).to_frame()
new_global_cases= new_global_cases.iloc[2:-1]
new_global_cases['New Daily Cases'] = new_global_cases[0].diff(1)
new_global_cases = new_global_cases.rename(columns={0: 'Total Confirmed Cases'})
days = range(0,len(new_global_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_global_cases['New Daily Cases'], linewidth=2)
plt.title('New Daily Confirmed Cases (Worldwide)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Cases')
plt.xticks(days, days)
plt.show()
new_global_deaths = deaths_global.sum(axis=0).to_frame()
new_global_deaths = new_global_deaths.iloc[2:-1]
new_global_deaths['New Daily Deaths'] = new_global_deaths[0].diff(1)
new_global_deaths = new_global_deaths.rename(columns={0: 'Total Confirmed Deaths'})
days = range(0,len(new_global_deaths),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_global_deaths['New Daily Deaths'], linewidth=2)
plt.title('New Daily Confirmed Deaths (Worldwide)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Deaths')
plt.xticks(days, days)
plt.show()
new_canada_cases = canada_confirmed_cases.sum(axis=0).to_frame()
new_canada_cases = new_canada_cases.iloc[2:-1]
new_canada_cases['New Daily Cases'] = new_canada_cases[0].diff(1)
new_canada_cases = new_canada_cases.rename(columns={0: 'Total Confirmed Cases'})
days = range(0,len(new_canada_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_canada_cases['New Daily Cases'], linewidth=2)
plt.title('New Daily Confirmed Cases (Canada)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Cases')
plt.xticks(days, days)
plt.show()
new_canada_deaths = canada_confirmed_deaths.sum(axis=0).to_frame()
new_canada_deaths = new_canada_deaths.iloc[2:-1]
new_canada_deaths['New Daily Deaths'] = new_canada_deaths[0].diff(1)
new_canada_deaths = new_canada_deaths.rename(columns={0: 'Total Confirmed Deaths'})
days = range(0,len(new_canada_deaths),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_canada_deaths['New Daily Deaths'], linewidth=2)
plt.title('New Daily Confirmed Deaths (Canada)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Deaths')
plt.xticks(days, days)
plt.show()
new_uk_cases = uk_confirmed_cases.sum(axis=0).to_frame()
new_uk_cases = new_uk_cases.iloc[2:-1]
new_uk_cases['New Daily Cases'] = new_uk_cases[0].diff(1)
new_uk_cases = new_uk_cases.rename(columns={0: 'Total Confirmed Cases'})
days = range(0,len(new_uk_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_uk_cases['New Daily Cases'], linewidth=2)
plt.title('New Daily Confirmed Cases (UK)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Cases')
plt.xticks(days, days)
plt.show()
new_uk_deaths = uk_confirmed_deaths.sum(axis=0).to_frame()
new_uk_deaths = new_uk_deaths.iloc[2:-1]
new_uk_deaths['New Daily Deaths'] = new_uk_deaths[0].diff(1)
new_uk_deaths = new_uk_deaths.rename(columns={0: 'Total Confirmed Deaths'})
days = range(0,len(new_uk_deaths),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_uk_deaths['New Daily Deaths'], linewidth=2)
plt.title('New Daily Confirmed Deaths (UK)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Deaths')
plt.xticks(days, days)
plt.show()
new_us_cases = us_confirmed_cases.sum(axis=0).to_frame()
new_us_cases = new_us_cases.iloc[2:-1]
new_us_cases['New Daily Cases'] = new_us_cases[0].diff(1)
new_us_cases = new_us_cases.rename(columns={0: 'Total Confirmed Cases'})
days = range(0,len(new_us_cases),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_us_cases['New Daily Cases'], linewidth=2)
plt.title('New Daily Confirmed Cases (US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Cases')
plt.xticks(days, days)
plt.show()
new_us_deaths = us_confirmed_deaths.sum(axis=0).to_frame()
new_us_deaths = new_us_deaths.iloc[2:-1]
new_us_deaths['New Daily Deaths'] = new_us_deaths[0].diff(1)
new_us_deaths = new_us_deaths.rename(columns={0: 'Total Confirmed Deaths'})
days = range(0,len(new_us_deaths),10)

ax = plt.figure(figsize=(16,6))
plt.plot(new_us_deaths['New Daily Deaths'], linewidth=2)
plt.title('New Daily Confirmed Deaths (US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Deaths')
plt.xticks(days, days)
plt.show()
days = range(0,len(new_canada_cases),10)
plt.figure(figsize=(16,6))
plt.plot(new_canada_cases['New Daily Cases'], linewidth=2)
plt.plot(new_uk_cases['New Daily Cases'], linewidth=2)
plt.plot(new_us_cases['New Daily Cases'], linewidth=2)
plt.title('Comparison of New Daily Cases (Canada, UK and US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Cases')
plt.legend(['Canada', 'UK', 'US'])
plt.xticks(days, days)
plt.show()
days = range(0,len(new_canada_deaths),10)
plt.figure(figsize=(16,6))
plt.plot(new_canada_deaths['New Daily Deaths'], linewidth=2)
plt.plot(new_uk_deaths['New Daily Deaths'], linewidth=2)
plt.plot(new_us_deaths['New Daily Deaths'], linewidth=2)
plt.title('Comparison of New Daily Deaths (Canada, UK and US)')
plt.xlabel('Days since first confirmed case')
plt.ylabel('New Daily Deaths')
plt.legend(['Canada', 'UK', 'US'])
plt.xticks(days, days)
plt.show()
confirmed_us_states = pd.read_csv('../input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
confirmed_us_states = confirmed_us_states.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Lat', 'Long_'])
confirmed_us_states = confirmed_us_states.groupby('Province_State').sum().reset_index()
confirmed_us_states['Total Confirmed'] = confirmed_us_states.iloc[:,-1]
confirmed_us_states.head()
deaths_us_states = pd.read_csv('../input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
deaths_us_states = deaths_us_states.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Lat', 'Long_'])
deaths_us_states = deaths_us_states.groupby('Province_State').sum().reset_index()
deaths_us_states['Total Deaths'] = deaths_us_states.iloc[:,-1]
deaths_us_states.head()
latest_us_states_confirmed = confirmed_us_states[['Province_State', 'Total Confirmed']]
latest_us_states_deaths = deaths_us_states[['Province_State', 'Population', 'Total Deaths']]
latest_us_states = latest_us_states_confirmed.merge(latest_us_states_deaths, how='left')
latest_us_states = latest_us_states[['Province_State', 'Population', 'Total Confirmed', 'Total Deaths']]
latest_us_states['Mortality Rate (%)'] = (latest_us_states['Total Deaths'] / latest_us_states['Total Confirmed'] * 100).round(2).fillna(0)
latest_us_states['Cases per 100,000'] = (latest_us_states['Total Confirmed'] / latest_us_states['Population'] * 100000).round(0)
latest_us_states['Deaths per 100,000'] = (latest_us_states['Total Deaths'] / latest_us_states['Population'] * 100000).round(0)
latest_us_states = latest_us_states.drop(latest_us_states[(latest_us_states.Province_State == 'American Samoa')
                                                         | (latest_us_states.Province_State == 'Diamond Princess')
                                                         | (latest_us_states.Province_State == 'Grand Princess')
                                                         | (latest_us_states.Province_State == 'Guam')
                                                         | (latest_us_states.Province_State == 'Northern Mariana Islands')
                                                         | (latest_us_states.Province_State == 'Puerto Rico')
                                                         | (latest_us_states.Province_State == 'Virgin Islands')].index).reset_index(drop=True)
latest_us_states.to_csv('latest_us_states.csv')
latest_us_states.head()
us_states_confirmed_sorted = latest_us_states.sort_values('Total Confirmed')
confirmed_sorted = us_states_confirmed_sorted['Total Confirmed']
states = us_states_confirmed_sorted['Province_State']

plt.figure(figsize=(12,10))
ax = plt.subplot()
plt.barh(states, confirmed_sorted, color='red', edgecolor='black')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)
plt.title('Confirmed Cases in US States', size=16, style='oblique')
plt.xlabel('Confirmed Cases', size=12)
plt.ylabel('US State', size=12)
plt.show()
us_states_deaths_sorted = latest_us_states.sort_values('Total Deaths')
deaths_sorted = us_states_deaths_sorted['Total Deaths']
states = us_states_confirmed_sorted['Province_State']

plt.figure(figsize=(12,10))
ax = plt.subplot()
plt.barh(states, deaths_sorted, color='red', edgecolor='black')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)
plt.title('Confirmed Deaths in US States', size=16, style='oblique')
plt.xlabel('Confirmed Deaths', size=12)
plt.ylabel('US State', size=12)
plt.show()
us_states_mortality_sorted = latest_us_states.sort_values('Mortality Rate (%)')
mortality_sorted = us_states_mortality_sorted['Mortality Rate (%)']
states = us_states_mortality_sorted['Province_State']

plt.figure(figsize=(12,10))
ax = plt.subplot()
plt.barh(states, mortality_sorted, color='red', edgecolor='black')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)
plt.title('Mortality Rate (%) in US States', size=16, style='oblique')
plt.xlabel('Mortality Rate (%)', size=12)
plt.ylabel('US State', size=12)
plt.show()
us_states_cases_population_sorted = latest_us_states.sort_values('Cases per 100,000')
cases_population_sorted = us_states_cases_population_sorted['Cases per 100,000']
states = us_states_cases_population_sorted['Province_State']

plt.figure(figsize=(12,10))
ax = plt.subplot()
plt.barh(states, cases_population_sorted, color='red', edgecolor='black')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)
plt.title('Cases per 100,000 Population in US States', size=16, style='oblique')
plt.xlabel('Cases per 100,000 Population', size=12)
plt.ylabel('US State', size=12)
plt.show()
us_states_deaths_population_sorted = latest_us_states.sort_values('Deaths per 100,000')
deaths_population_sorted = us_states_deaths_population_sorted['Deaths per 100,000']
states = us_states_deaths_population_sorted['Province_State']

plt.figure(figsize=(12,10))
ax = plt.subplot()
plt.barh(states, deaths_population_sorted, color='red', edgecolor='black')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)
plt.title('Deaths per 100,000 Population in US States', size=16, style='oblique')
plt.xlabel('Deaths per 100,000 Population', size=12)
plt.ylabel('US State', size=12)
plt.show()
new_confirmed_global = confirmed_global.set_index('Country/Region')
countries = list(new_confirmed_global.index)
avg_infection_rates = []
for country in countries:
    avg_infection_rates.append(new_confirmed_global.loc[country].diff().mean())
new_confirmed_global['avg_infection_rate'] = avg_infection_rates
country_avg_infection_rates = pd.DataFrame(new_confirmed_global['avg_infection_rate'])
country_avg_infection_rates.head()
world_happiness_report = pd.read_csv('../input/world-happiness-report/world_happiness_report.csv')
columns_to_drop = ['Score', 'Generosity', 'Perceptions of corruption']
world_happiness_report.drop(columns=columns_to_drop, inplace=True)
world_happiness_report.set_index(['Country'], inplace=True)
world_happiness_report.head()
happiness_data = world_happiness_report.join(country_avg_infection_rates).copy()
happiness_data.head()
happiness_data.corr()
plt.figure(figsize=(12,10))

ax1 = plt.subplot(2, 2, 1)
x1 = happiness_data['GDP per capita']
y1 = happiness_data['avg_infection_rate']
sns.regplot(x1, np.log(y1))

ax2 = plt.subplot(2, 2, 2)
x2 = happiness_data['Social support']
y2 = happiness_data['avg_infection_rate']
sns.regplot(x2, np.log(y2))

ax3 = plt.subplot(2, 2, 3)
x3 = happiness_data['Healthy life expectancy']
y3 = happiness_data['avg_infection_rate']
sns.regplot(x3, np.log(y3))

ax4 = plt.subplot(2, 2, 4)
x4 = happiness_data['Freedom to make life coaches']
y4 = happiness_data['avg_infection_rate']
sns.regplot(x4, np.log(y4))
new_deaths_global = deaths_global.set_index('Country/Region')
countries = list(new_deaths_global.index)
avg_death_rates = []
for country in countries:
    avg_death_rates.append(new_deaths_global.loc[country].diff().mean())
new_deaths_global['avg_death_rate'] = avg_death_rates
country_avg_death_rates = pd.DataFrame(new_deaths_global['avg_death_rate'])
country_avg_death_rates.sort_values(['avg_death_rate'], ascending=True).head()
happiness_data2 = world_happiness_report.join(country_avg_death_rates).copy()
happiness_data2.head()
happiness_data2.corr()
plt.figure(figsize=(12,10))

ax1 = plt.subplot(2, 2, 1)
x1 = happiness_data2['GDP per capita']
y1 = happiness_data2['avg_death_rate']
result1 = np.where(y1 > 0.0000000001, y1, -10)
sns.regplot(x1, np.log(result1, out=result1, where=result1 > 0))

ax2 = plt.subplot(2, 2, 2)
x2 = happiness_data2['Social support']
y2 = happiness_data2['avg_death_rate']
result2 = np.where(y2 > 0.0000000001, y2, -10)
sns.regplot(x2, np.log(result2, out=result2, where=result2 > 0))

ax3 = plt.subplot(2, 2, 3)
x3 = happiness_data2['Healthy life expectancy']
y3 = happiness_data2['avg_death_rate']
result3 = np.where(y3 > 0.0000000001, y3, -10)
sns.regplot(x3, np.log(result3, out=result3, where=result3 > 0))

ax4 = plt.subplot(2, 2, 4)
x4 = happiness_data2['Freedom to make life coaches']
y4 = happiness_data2['avg_death_rate']
result4 = np.where(y4 > 0.0000000001, y4, -10)
sns.regplot(x4, np.log(result4, out=result4, where=result4 > 0))