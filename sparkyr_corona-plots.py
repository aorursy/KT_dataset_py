import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
df = df.fillna(value='')

df['Location'] = df['Country/Region'] + ' ' + df['Province/State']

df['Last Update'] = pd.to_datetime(df['Last Update'])

df = df.sort_values(by='Last Update')

df.head()
df = df.sort_values(by='Last Update')

df['Active'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
min_active = 50

df['ActiveDelta'] = 0

for location in df['Location'].unique():

    l = df['Location'] == location

    dfl = df[l]

    df.loc[l,'ActiveDelta'] = dfl['Active'].diff()/(dfl['Last Update'].diff() / pd.to_timedelta(1, unit='D'))



#df['ActiveDelta'] = df.groupby('Location').transform(lambda df: df['Active'].diff()/(df['Last Update'].diff() / pd.to_timedelta(1, unit='D')))



df['Alpha'] = df['ActiveDelta'] / df['Active']

df['Alpha'] = np.where(df['Active'] >= min_active, df['Alpha'], np.NaN)

df.head()
std = 24

min_active = 50

def calc_alpha_smooth(df):

    df = df.set_index('Last Update').resample('h').mean().interpolate(method='linear')

    df = df.rolling(window=2*std, win_type='gaussian', center=True).mean(std=std)

    active = df['Active']

    active_dt = active.diff() / (active.index.to_series().diff() / pd.to_timedelta(1, unit='D'))

    alpha = active_dt / active

    alpha[active < min_active] = np.NaN

    return alpha

    

alpha = df.groupby(['Country/Region','Location']).apply(calc_alpha_smooth).to_frame('Alpha').reset_index().dropna()

alpha.head()
fig = px.line(df, x='Last Update', y='Confirmed', color='Country/Region', line_group='Location', log_y=True, range_y=[10, 100000]).update_traces(mode='lines+markers')

fig.update_layout(height=1000)
fig = px.line(df, x='Last Update', y='Active', color='Country/Region', line_group='Location', log_y=True, range_y=[10, 100000]).update_traces(mode='lines+markers')

fig.update_layout(height=1000)
fig = px.line(df, x='Last Update', y='Alpha', color='Country/Region', line_group='Location').update_traces(mode='lines+markers')

fig.update_layout(height=1000)
fig = px.line(alpha, x='Last Update', y='Alpha', color='Country/Region', line_group='Location')

fig.update_layout(height=1000)