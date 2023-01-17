# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.graph_objects as go
cases = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

print(cases.shape)



beds = pd.read_csv("/kaggle/input/hospital-beds-data-india/hospital_beds.csv")

print(beds.shape)
cases['State/UnionTerritory'].unique()
cases['State/UnionTerritory'].replace({"Telengana" : "Telangana", "Telengana***" : "Telangana",

                                        "Telangana***" : "Telangana"}, inplace = True)



cases['State/UnionTerritory'].replace({"Daman & Diu" : "Dadra and Nagar Haveli and Daman and Diu",

                                          "Dadar Nagar Haveli" : "Dadra and Nagar Haveli and Daman and Diu"},

                                         inplace = True)

cases = cases[(cases['State/UnionTerritory'] != 'Unassigned') &

                    (cases['State/UnionTerritory'] != 'Cases being reassigned to states')]

cases['State/UnionTerritory'].unique()
cases.Date = pd.to_datetime(cases.Date, dayfirst=True)



cases.drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis = 1, inplace=True)

cases.head()
daily_cases = cases.groupby('Date').sum().reset_index()

daily_cases['Active'] = 1



for val in daily_cases.index:

    if val != 0:

        daily_cases['Active'].loc[val] = daily_cases['Confirmed'].loc[val] - daily_cases['Cured'].loc[val-1] - daily_cases['Deaths'].loc[val-1]

    

daily_cases
fig = go.Figure()

fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Active, name = 'Active Cases'))



fig.update_layout(title = 'Daily Active Cases', xaxis_title = 'Time', yaxis_title = 'Count (in lakhs)')

fig.show()
state_daily_cases = cases.sort_values(by=['State/UnionTerritory', 'Date']).reset_index(drop=True)

state_daily_cases['ActiveCases'] = 0



for st in sorted(cases['State/UnionTerritory'].unique()):

    df = state_daily_cases[state_daily_cases['State/UnionTerritory'] == st]

    for i in df.index:

        conf = state_daily_cases['Confirmed'].iloc[i]

        rec = state_daily_cases['Cured'].iloc[i-1]

        death = state_daily_cases['Deaths'].iloc[i-1]

            

        state_daily_cases['ActiveCases'].iloc[i] = conf - rec - death

    state_daily_cases['ActiveCases'].iloc[df.index[0]] = state_daily_cases['Confirmed'].iloc[df.index[0]]

fig = go.Figure()

for st in state_daily_cases['State/UnionTerritory'].unique():

    df = state_daily_cases[state_daily_cases['State/UnionTerritory'] == st]

    fig.add_trace(go.Scatter(x = df['Date'], y = df['ActiveCases'], name = st))



fig.update_layout(title = 'Daily Active Cases', xaxis_title = 'Time', yaxis_title = 'Count (in lakhs)')

fig.show()
total = beds.iloc[35]

print("Total beds available all over India", total)

beds.drop([35], inplace=True)

beds.head()
beds_per_state = state_daily_cases.set_index('State/UnionTerritory').join(beds.set_index('States'))

beds_per_state['AvailableBeds'] = beds_per_state['Total hospital beds'] - beds_per_state['ActiveCases']

beds_per_state
beds_df = beds_per_state[['Date', 'AvailableBeds']]

beds_df = pd.pivot_table(beds_per_state, values = 'AvailableBeds', index = 'Date',

                               columns = beds_per_state.index)

for st in beds_df.columns:

    val = beds[beds['States'] == st]['Total hospital beds']

    beds_df[st].fillna(int(val), inplace=True)

    

beds_df.head()



# If we want the data to be in csv format then,

# beds_df = beds_df.to_csv("beds_data.csv")
fig = go.Figure()

for col in beds_df.columns:

    fig.add_trace(go.Scatter(x = beds_df.index, y = beds_df[col], name = col))



fig.update_layout(title = 'Number of available beds statewise', yaxis_title = 'Number of available beds')

fig.show()