# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pylab import plt

import requests

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
#states_daily_df = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')
#states_daily_df = pd.read_csv('../input/covid19inusadailydirect/daily.csv')
states_daily_df = pd.read_csv('http://covidtracking.com/api/states/daily.csv')

states_daily_df
#state_population_df = pd.read_csv("../input/us-census-2019-population-estimate/SCPRC-EST2019-18+POP-RES.csv")
#state_population_df = state_population_df[['NAME', 'POPESTIMATE2019']]
#state_population_df = state_population_df.rename(columns={'NAME': 'StateName', 'POPESTIMATE2019': 'EstimatedPopulation2019'})
#state_population_df.head()
## Scrape a population estimate for us states and territories.
tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population")
state_population_df = tables[0]
state_population_df = state_population_df[['State', 'Census population']]
state_population_df.columns = state_population_df.columns.to_flat_index()
state_population_df = state_population_df[[('State', 'State'), state_population_df.columns[2]]]
state_population_df.columns = state_population_df.columns = ['StateName', 'EstimatedPopulation']
state_population_df['EstimatedPopulation'] = state_population_df['EstimatedPopulation'].str.replace("(,)|(\[\d+\])","").astype(int)
state_population_df['StateName'] = state_population_df['StateName'].str.replace('U.S. ', '')
state_population_df
## Scrape a mapping of territories to postal state/territory codes
#state_abbrev_df = pd.read_csv("../input/state-abbreviations/state_abbrev.csv")[['State', 'Abbreviation']]
#state_abbrev_df = state_abbrev_df.rename(columns={'Abbreviation': 'StateAbbreviation', 'State': 'StateName'})
tables = pd.read_html(requests.get("https://pe.usps.com/text/pub28/28apb.htm",
                                   headers={'User-agent': 'Mozilla/5.0'}).text,
                     attrs={"id": "ep18684"},
                     header=0)
state_abbrev_df = tables[0]
state_abbrev_df = state_abbrev_df.rename(columns={'Abbreviation': 'StateAbbreviation', 'State/Possession': 'StateName'})
state_abbrev_df = state_abbrev_df.set_index('StateName')
state_abbrev_df
state_population_df = state_population_df.join(state_abbrev_df, 'StateName').set_index('StateAbbreviation')
state_population_df.head()
state_population_df
states_daily_df = states_daily_df.join(state_population_df, 'state')
states_daily_df
#correlatesofstatepolicyprojectv2_2_df = pd.read_csv("../input/msu-correlates-of-state-policy-v22/correlatesofstatepolicyprojectv2_2.csv")
#state_political_control_2016_df = correlatesofstatepolicyprojectv2_2_df[correlatesofstatepolicyprojectv2_2_df['year'] == 2016][['st', 'ranney4_control']].set_index('st')
#states_daily_df = states_daily_df.join(state_political_control_2016_df, 'state')
#states_daily_df
states_daily_df[states_daily_df['state'] == 'WV']
states_daily_df['total_pct'] = states_daily_df['total'] / states_daily_df['EstimatedPopulation'] * 100
states_daily_df['positive_pct'] = states_daily_df['positive'] / states_daily_df['EstimatedPopulation'] * 100
states_daily_df['death_pct'] = states_daily_df['death'] / states_daily_df['EstimatedPopulation'] * 100

states_daily_df['date'] = pd.to_datetime(states_daily_df['date'], format='%Y%m%d')
states_daily_df.set_index('date')

latest_test_date = states_daily_df['date'].max()
latest_test_rates_df = states_daily_df[states_daily_df['date'] == latest_test_date].sort_values(by='total_pct', ascending=False).reset_index()[['date', 'state', 'positive', 'negative','pending','death','total','total_pct','positive_pct','death_pct','EstimatedPopulation']]
latest_test_rates_df
ax = states_daily_df[states_daily_df['date'] == latest_test_date].sort_values(by='total_pct', ascending=False)[['state', 'total_pct']].set_index('state').plot.bar(
    figsize = (12,5),
)
ax.set_title("Percentage of population tested by " + str(latest_test_date))
ax.set_xlabel("State")
ax.set_ylabel("% tested")
_ = ax
ax = states_daily_df[states_daily_df['date'] == latest_test_date].sort_values(by='total_pct', ascending=False)[['state', 'positive_pct']].set_index('state').plot.bar(
    figsize = (12,5),
)
ax.set_title("Percentage of population testing positive " + str(latest_test_date))
ax.set_xlabel("State")
ax.set_ylabel("% tested positive")
_ = ax
states_daily_df[states_daily_df['state'] == 'VT']
ax = states_daily_df[states_daily_df['state'] == 'VT'][['date','positive_pct']].set_index('date').plot.line()
ax.set_ylim(0, 0.1)
ax.set_title('Positive Percentage of Population for VT')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'VT'][['date','positive']].set_index('date').plot.line()
ax.set_title('Positive Count of Population for VT')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'NY'][['date', 'positive_pct']].set_index('date').plot.line()
#ax.set_ylim(0, 1)
ax.set_title('Tested Positive Percentage of Population for NY')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'WA'][['date', 'positive_pct']].set_index('date').plot.line()
ax.set_ylim(0, 0.2)
ax.set_title('Tested Positive Percentage of Population for WA')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'DC'][['date', 'positive_pct']].set_index('date').plot.line()
ax.set_ylim(0, 0.2)
ax.set_title('Tested Positive Percentage of Population for DC')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'NJ'][['date', 'positive_pct']].set_index('date').plot.line()
ax.set_ylim(0, 0.3)
ax.set_title('Tested Positive Percentage of Population for NJ')
_=ax
ax = states_daily_df[states_daily_df['state'] == 'GA'][['date', 'positive_pct']].set_index('date').plot.line()
ax.set_ylim(0, 0.1)
ax.set_title('Tested Positive Percentage of Population for GA')
ax.set_xlabel("Date")
ax.set_ylabel("% tested positive")
_=ax
def plot_state(abbrev):
    data = states_daily_df[states_daily_df['state'] == abbrev][['date', 'positive_pct']].set_index('date')
    line, = plt.plot(data)
    if( max(data['positive_pct']) > 0.1):
        line.set_label(abbrev + "  " + f"{data['positive_pct'].round(2).max()}")
    
plt.figure(figsize=(20,10))
plt.ylabel("% of Population Tested Positive")
plt.xlabel("Date")
plt.title('Percentage of Population Testing Positive')
#state_abbrev_df['StateAbbreviation'].map( plot_state, na_action='ignore')
states_daily_df[states_daily_df['date'] == latest_test_date].sort_values(by='positive_pct', ascending=False)['state'].map( plot_state, na_action='ignore')
_ = plt.legend()
states_daily_df[states_daily_df['date'] == latest_test_date].sort_values(by='positive_pct', ascending=False)
plt.figure(figsize=(20,10))
plt.yscale('log')
plt.ylabel("Number of Positive Tests (log scale)")
plt.xlabel("Date")
plt.title('Number of Positive Tests Over Time (log scale)')
_ = state_abbrev_df['StateAbbreviation'].map( lambda it: plt.plot(states_daily_df[states_daily_df['state'] == it][['date', 'positive']].set_index('date')), na_action='ignore')

plt.figure(figsize=(20,10))
plt.yscale('log')
plt.ylabel("Percentage of Population Testing Positive Over Time (log scale)")
plt.xlabel("Date")
plt.title('Percentage of Population Testing Positive (log scale)')
_ = state_abbrev_df['StateAbbreviation'].map( lambda it: plt.plot(states_daily_df[states_daily_df['state'] == it][['date', 'positive_pct']].set_index('date')), na_action='ignore')

def plot_test_for_state(abbrev):
    plt.figure(figsize=(20,10))
    plt.title('Test results for % of population for ' + abbrev)
    plt.ylabel("% of Population")
    plt.xlabel("Date")
    plt.yscale('log', basey=10)
    plt.ylim( (10**-5,10**1) )

    data = states_daily_df[states_daily_df['state'] == abbrev][['date', 'total_pct']].set_index('date')
    line, = plt.plot(data)
    line.set_label(abbrev + " Tested %")
    
    data = states_daily_df[states_daily_df['state'] == abbrev][['date', 'positive_pct']].set_index('date')
    line, = plt.plot(data)
    line.set_label(abbrev + " Positive %")
    
    data = states_daily_df[states_daily_df['state'] == abbrev][['date', 'death_pct']].set_index('date')
    line, = plt.plot(data)
    line.set_label(abbrev + " Death %")
    
    plt.legend()
    
    return plt
_ = plot_test_for_state("NY")
_ = plot_test_for_state('WA')
_ = plot_test_for_state('VT')
_ = plot_test_for_state('LA')
_ = plot_test_for_state('NJ')
