import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import datasets



individuals_df = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv", index_col=0)

covid_df = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv", index_col=0).drop('Time', axis=1)

covid_features_df = pd.read_csv("/kaggle/input/features-covid/statewise_features.csv")
individuals_df.head()
individuals_df.info()
individuals_df['current_status'].unique()
# Time span of this dataset

individuals_df['diagnosed_date'] = pd.to_datetime(individuals_df['diagnosed_date'], format="%d/%m/%Y")

individuals_df['diagnosed_date'].min(), individuals_df['diagnosed_date'].max()
# Casting age to numeric 



cases_data = individuals_df[pd.to_numeric(individuals_df['age'], errors='coerce').notnull()]

cases_data['age'] = pd.to_numeric(cases_data['age'])

cases_data['age'].describe()
plt.figure(figsize=(6,4))

sns.kdeplot(cases_data['age'])

plt.title("Distribution of age for all cases")

plt.xlabel("Age")

plt.savefig("./age_distn_all.png")

plt.show()
plt.figure(figsize=(6,4))

sns.kdeplot(cases_data[cases_data['current_status']=='Deceased']['age'])

plt.title("Distribution of age for deceased")

plt.xlabel("Age")

plt.savefig("./deceased_age_dist.png")

plt.show()
plt.figure(figsize=(10,6))

for state in cases_data['detected_state'].unique():

    subset = cases_data[cases_data['detected_state']==state]

    if subset.count()['current_status'] > 1:

        sns.kdeplot(subset['age'])

plt.title("Statewise distribution of Age")

plt.xlabel("age")

plt.legend(cases_data['detected_state'].unique())

plt.show()
plt.figure(figsize=(10,6))

for state in cases_data['detected_state'].unique():

    subset = cases_data[cases_data['detected_state']==state]

    deceased_subset = subset[subset['current_status']=='Deceased']

    if deceased_subset.count()['current_status'] > 1:

        sns.kdeplot(subset[subset['current_status']=='Deceased']['age'])

plt.title("Statewise distribution of age w.r.t deceased ")

plt.xlabel("age")

plt.legend(cases_data['detected_state'].unique())

plt.savefig("./deceased_age_dist_statewise.png")

plt.show()
plt.figure(figsize=(10,6))

for state in cases_data['detected_state'].unique():

    subset = cases_data[cases_data['detected_state']==state]

    deceased_subset = subset[subset['current_status']=='Recovered']

    if deceased_subset.count()['current_status'] > 1:

        sns.kdeplot(subset[subset['current_status']=='Recovered']['age'])

plt.title("Statewise distribution of Age for Recovered")

plt.xlabel("age")

plt.legend(cases_data['detected_state'].unique())

plt.savefig("./recovered_age_dist_statewise.png")

plt.show()
covid_df['Date'] = pd.to_datetime(covid_df['Date'], format="%d/%m/%y")



covid_df['Date'].min(), covid_df['Date'].max()
plt.figure(figsize=(13,6))

for state in covid_df['State/UnionTerritory'].unique():

    covid_df[covid_df['State/UnionTerritory']==state].set_index('Date')['Confirmed'].plot()

plt.legend(covid_df['State/UnionTerritory'].unique())

plt.show()
covid_df.count()
covid_daily_df = None

for state in covid_df['State/UnionTerritory'].unique():

    covid_data_state = covid_df[covid_df['State/UnionTerritory']==state]

    covid_data_state['previous_day'] = covid_data_state['Confirmed'].shift(1)

    covid_data_state['new_cases'] = covid_data_state['Confirmed'] - covid_data_state['previous_day']



    covid_data_state['previous_day'] = covid_data_state['Deaths'].shift(1)

    covid_data_state['fatalities'] = covid_data_state['Deaths'] - covid_data_state['previous_day']



    covid_data_state['previous_day'] = covid_data_state['Cured'].shift(1)

    covid_data_state['recoveries'] = covid_data_state['Cured'] - covid_data_state['previous_day']



    covid_data_state = covid_data_state.drop('previous_day',axis=1)

    covid_daily_df = pd.concat([covid_daily_df, covid_data_state], axis=0)

    

covid_daily_df.set_index('Date', inplace=True)
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in covid_daily_df['State/UnionTerritory'].unique():    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['new_cases'].plot()

    

    

plt.legend(covid_daily_df['State/UnionTerritory'].unique(), loc='upper left')

plt.title("Daily new cases")

plt.savefig("./daily_new_cases.png")

plt.show()
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in covid_daily_df['State/UnionTerritory'].unique():    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['recoveries'].plot()

    

plt.legend(covid_daily_df['State/UnionTerritory'].unique(), loc='upper left')

plt.title("Daily recoveries")

plt.savefig("./daily_recoveries.png")

plt.show()
states = covid_daily_df['State/UnionTerritory'].unique()
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in states[:10]:    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['fatalities'].plot()

    

plt.legend(states[:10], loc='upper left')

plt.title("Daily Fatalities")

plt.savefig("./fatalities.png")

plt.show()
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in states[10:20]:    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['fatalities'].plot()

plt.legend(states[10:20], loc='upper left')

plt.title("Fatalities")

plt.show()
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in states[20:30]:    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['fatalities'].plot()

plt.legend(states[20:30], loc='upper left')

plt.title("Fatalities")

plt.show()
samples = 0

labels = covid_daily_df.reset_index()['Date']

x_pos = np.arange(len(labels))

plt.figure(figsize=(13,6))



for state in states[30:]:    

    covid_daily_df[covid_daily_df['State/UnionTerritory']==state]['fatalities'].plot()

plt.legend(states[30:], loc='upper left')

plt.title("Fatalities")

plt.show()
covid_features_df.head()
plt.figure(figsize=(12,6))

covid_features_df.groupby("State").max().sort_values("pop_density", ascending=False)['pop_density'].plot(kind='bar')

plt.title("Statewise Population density per Km2")

plt.show()