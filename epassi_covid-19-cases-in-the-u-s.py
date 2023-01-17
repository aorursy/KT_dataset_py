import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

raw_us_df = pd.read_csv('../input/nytimes-covid19-data/us.csv')
raw_states_df = pd.read_csv('../input/nytimes-covid19-data/us-states.csv')
raw_counties_df = pd.read_csv('../input/nytimes-covid19-data/us-counties.csv')

dates = pd.date_range(dt.date(2020, 1, 1), dt.datetime.today()-dt.timedelta(days=2)).tolist()
dates_str = [date.strftime('%Y-%m-%d') for date in dates]

def state_case_count(state, date):
    value = raw_states_df['cases'][raw_states_df['state'] == state][raw_states_df['date'] == date]
    return 0 if value.empty else value.iloc[0]

def state_death_count(state, date):
    value = raw_states_df['deaths'][raw_states_df['state'] == state][raw_states_df['date'] == date]
    return 0 if value.empty else value.iloc[0]

def county_case_count(county, date):
    value = raw_counties_df['cases'][raw_counties_df['county'] == county][raw_counties_df['date'] == date]
    return 0 if value.empty else value.iloc[0]

def county_death_count(county, date):
    value = raw_counties_df['deaths'][raw_counties_df['county'] == county][raw_counties_df['date'] == date]
    return 0 if value.empty else value.iloc[0]
raw_us_df.tail()
raw_states_df.tail()
raw_counties_df.tail()
plt.figure(figsize=(20, 5))
plt.plot(raw_us_df['date'], raw_us_df['cases'])
plt.plot(raw_us_df['date'], raw_us_df['deaths'])
plt.xticks(raw_us_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases & Deaths in Millions')
plt.legend(['Cases', 'Deaths'])
plt.title('Cumulative Cases & Deaths Nationwide')
plt.show()
top_count = 10
cases_by_state_df = raw_states_df.drop(['fips'], axis=1)
cases_by_state_df = cases_by_state_df.groupby('state').max().sort_values(by=['cases'], ascending=False).reset_index()
cases_by_state_df.head(top_count)
states_with_most_cases_df = cases_by_state_df.iloc[:top_count, :];
plt.figure(figsize=(20, 7))
plt.barh(np.arange(top_count)+0.125, states_with_most_cases_df['cases'].iloc[::-1], height=0.25)
plt.barh(np.arange(top_count)-0.125, states_with_most_cases_df['deaths'].iloc[::-1], height=0.25)
plt.yticks(np.arange(top_count), states_with_most_cases_df['state'].iloc[::-1])
plt.xlabel('Cases & Deaths')
plt.legend(['Cases', 'Deaths'])
plt.title('Highest Cumulative Case Counts Nationwide')
plt.show()
usa_df = pd.DataFrame({'date': dates_str})
most_affected_states = cases_by_state_df['state'].head(top_count).to_list()

for state in most_affected_states:
    usa_df[state] = [state_case_count(state, date) for date in dates_str]
    
usa_df.tail()
plt.figure(figsize=(20, 5))

for state in most_affected_states:
    plt.plot(usa_df['date'], usa_df[state])
    
plt.xticks(usa_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases')
plt.legend(most_affected_states)
plt.title('Cumulative Cases in Most Affected States')
plt.show()
state = 'California'
state_cases_df = raw_states_df[raw_states_df['state'] == state].groupby('date').sum().reset_index()
state_cases_df.tail()
plt.figure(figsize=(20, 5))
plt.plot(state_cases_df['date'], state_cases_df['cases'], color=(0.2, 0.5, 1))
plt.plot(state_cases_df['date'], state_cases_df['deaths'], color=(1, 0.5, 0))
plt.xticks(state_cases_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases & Deaths')
plt.legend(['Cases', 'Deaths'])
plt.title('Cumulative Cases & Deaths in California')
plt.show()
major_counties = ['Los Angeles', 'San Diego', 'Santa Clara', 'San Francisco', 'Sacramento']
major_counties_df = pd.DataFrame({'date': dates_str})
for county in major_counties:    
    major_counties_df[county] = [county_case_count(county, date) for date in dates_str]

major_counties_df.tail()
plt.figure(figsize=(20, 5))

for county in major_counties:
    plt.plot(major_counties_df['date'], major_counties_df[county])
    
plt.xticks(major_counties_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases')
plt.legend(major_counties)
plt.title('Cumulative Cases & Deaths in Major California Counties')
plt.show()
high_case_county = 'Los Angeles'
ca_minus_high_case_county_df = pd.DataFrame()
ca_minus_high_case_county_df['date'] = dates_str
ca_minus_high_case_county_df['Total Cases'] = [state_case_count('California', date) for date in dates_str]
ca_minus_high_case_county_df[f'Cases minus {high_case_county} County'] = [state_case_count('California', date) - county_case_count(high_case_county, date) for date in dates_str]
ca_minus_high_case_county_df['Total Deaths'] = [state_death_count('California', date) for date in dates_str]
ca_minus_high_case_county_df[f'Deaths minus {high_case_county} County'] = [state_death_count('California', date) - county_death_count(high_case_county, date) for date in dates_str]

ca_minus_high_case_county_df.tail()
plt.figure(figsize=(20, 5))
plt.plot(ca_minus_high_case_county_df['date'], ca_minus_high_case_county_df['Total Cases'], color=(0.2, 0.5, 1, 0.3))
plt.plot(ca_minus_high_case_county_df['date'], ca_minus_high_case_county_df[f'Cases minus {high_case_county} County'], color=(0.2, 0.5, 1))
plt.plot(ca_minus_high_case_county_df['date'], ca_minus_high_case_county_df['Total Deaths'], color=(1, 0.5, 0, 0.3))
plt.plot(ca_minus_high_case_county_df['date'], ca_minus_high_case_county_df[f'Deaths minus {high_case_county} County'], color=(1, 0.5, 0))
plt.xticks(ca_minus_high_case_county_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases & Deaths')
plt.legend(['Total Cases', f'Cases minus {high_case_county} County', 'Total Deaths', f'Deaths minus {high_case_county} County'])
plt.title(f'Cumulative Cases & Deaths in California, minus {high_case_county} County')
plt.show()
bay_area_counties = ['Alameda', 'Contra Costa', 'Marin', 'San Francisco', 'San Mateo', 'Santa Clara']
bay_area_df = pd.DataFrame({'date': dates_str})
for county in bay_area_counties:    
    bay_area_df[county] = [county_case_count(county, date) for date in dates_str]

bay_area_df.tail()
plt.figure(figsize=(20, 5))

for county in bay_area_counties:
    plt.plot(bay_area_df['date'], bay_area_df[county])
    
plt.xticks(bay_area_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases')
plt.legend(bay_area_counties)
plt.title('Cumulative Cases in the SF Bay Area')
plt.show()
def county_cases_df(county):
    df = raw_counties_df[raw_counties_df['county'] == county].drop(['county', 'state', 'fips'], axis=1)
    df['new cases'] = df['cases'] - df['cases'].shift(1)
    df['new deaths'] = df['deaths'] - df['deaths'].shift(1)
    return df
    
san_francisco_df = county_cases_df('San Francisco')
san_francisco_df.tail()
def county_cumulative_cases_plt(county_df, title):
    plt.figure(figsize=(20, 5))
    plt.plot(san_francisco_df['date'], san_francisco_df['cases'])
    plt.plot(san_francisco_df['date'], san_francisco_df['deaths'])
    plt.xticks(san_francisco_df['date'][::5], rotation=45, ha='right')
    plt.ylabel('Cases & Deaths')
    plt.legend(['Cases', 'Deaths'])
    plt.title(title)
    return plt
    
county_cumulative_cases_plt(san_francisco_df, 'Cumulative Cases & Deaths in San Francisco').show()
def county_new_cases_plt(county_df, title):
    plt.figure(figsize=(20, 5))
    plt.plot(county_df['date'], county_df['new cases'])
    plt.plot(county_df['date'], county_df['new deaths'])
    plt.xticks(county_df['date'][::5], rotation=45, ha='right')
    plt.ylabel('New Cases & Deaths')
    plt.legend(['Cases', 'Deaths'])
    plt.title(title)
    return plt

county_new_cases_plt(san_francisco_df, 'New Cases & Deaths in San Francisco').show()
los_angeles_df = county_cases_df('Los Angeles')
county_new_cases_plt(los_angeles_df, 'New Cases & Deaths in LA County').show()
new_york_city_df = county_cases_df('New York City')
county_new_cases_plt(new_york_city_df, 'New Cases & Deaths in NYC').show()