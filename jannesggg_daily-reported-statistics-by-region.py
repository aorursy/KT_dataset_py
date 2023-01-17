# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_summary import DataFrameSummary



# viz tools

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

import cufflinks as cf

import plotly.express as px

import plotly.graph_objects as go

import plotly.tools as tls

import seaborn as sns

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, iplot_mpl



init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
kaggle_path = "/kaggle/input/sweden-covid19-dataset/"



death_df = pd.read_csv(kaggle_path + 'time_series_deaths-deaths.csv')

death_summary = DataFrameSummary(death_df)



case_df = pd.read_csv(kaggle_path + 'time_series_confimed-confirmed.csv')

case_summary = DataFrameSummary(death_df)

ts_death = death_df[[i for i in death_df.columns if i not in ['Display_Name', 'Population', 'Lat', 'Long', 'Diff', 'At_Hospital', 'At_ICU',

                       'FHM_Total', 'Region_Deaths', 'FHM_Deaths_Today', 'Hospital_Total']]].fillna(0)
ts_case = case_df[[i for i in death_df.columns if i not in ['Display_Name', 'Population', 'Lat', 'Long', 'Diff', 'At_Hospital', 'At_ICU',

                       'FHM_Total', 'Region_Deaths', 'FHM_Deaths_Today', 'Hospital_Total']]].fillna(0)
ts_cum_deaths = pd.concat([ts_death['Region'],ts_death.iloc[:,1:-2].cumsum(1)],1)

ts_cum_cases = pd.concat([ts_case['Region'], ts_case.iloc[:,1:-2].cumsum(1)],1)
ts_cum_deaths.head()
ts_cum_death_count = ts_cum_deaths.set_index('Region').T.drop(0, 1).drop('Total', 1)

ts_cum_cases_count = ts_cum_cases.set_index('Region').T.drop(0, 1).drop('Total', 1)

case_start = np.where(ts_cum_cases_count['Todays_Total'] > 100)[0][0]

death_start = np.where(ts_cum_cases_count['Todays_Total'] > 100)[0][0]
ts_case = ts_case.set_index('Region').T.drop(0, 1).drop('Total', 1)

fig = ts_case[case_start:].iloc[:,:-3].sum(1).rolling(7).mean().dropna().iplot(asFigure=True)

fig.update_layout(yaxis_type="log")

fig.show()
ts_death = ts_death.set_index('Region').T.drop(0, 1).drop('Total', 1)

fig = ts_death[death_start:].iloc[:,:-3].sum(1).rolling(7).mean().dropna().iplot(asFigure=True)

fig.update_layout(yaxis_type="log")

fig.show()
fig = ts_death[death_start:].iloc[:,:-3].rolling(7).mean().dropna().iplot(asFigure=True)

fig.update_layout(yaxis_type="log")

fig.show()
ts_cum_df = ts_cum_death_count[death_start:].T.reset_index().iloc[:,1:].divide(death_df['Population'].iloc[:-3]/100000, axis=0).T.set_axis(ts_cum_death_count[death_start:].T.index, axis=1, inplace=False).iloc[:,:-3]

fig = ts_cum_df.rolling(7).mean().dropna().iplot(asFigure=True)

fig.update_layout(yaxis_type="log")
fig = ts_death[death_start:].iloc[:,1:].divide(ts_case[death_start:].iloc[:,1:]).fillna(0).rolling(7, min_periods=1).mean().iplot(asFigure=True)

fig.update_layout(yaxis_type="log")
ts_death = ts_death[death_start:][ts_death['Region Stockholm'] > 0]



# Create the total score for each participant

totals = [i for i in ts_death['Todays_Total']]



# Create the percentage of the total score the pre_score value for each participant was

pre_rel = [i / j * 100 for i,j in zip(ts_death['Region Stockholm'], totals)]



# Create the percentage of the total score the mid_score value for each participant was

mid_rel = [i / j * 100 for  i,j in zip(ts_death['Västra Götalandsregionen'], totals)]



# Create the percentage of the total score the post_score value for each participant was

post_rel = [i / j * 100 for  i,j in zip(ts_death['Region Skåne'], totals)]



# Create the percentage of the total score the post_score value for each participant was



rest = [i / j * 100 for  i,j in zip(np.sum([ts_death[i] for i in ts_death.columns if i not in ['Region Stockholm', 'Västra Götalandsregionen', 'Region Skåne', 'Todays_Total']], 0), totals)]



import plotly.graph_objects as go

x=ts_death.index



fig = go.Figure(data=[

    go.Bar(name='Region Stockholm', x=x, y=pre_rel),

    go.Bar(name='Västra Götalandsregionen', x=x, y=mid_rel),

    go.Bar(name="Region Skåne", x=x, y=post_rel),

    go.Bar(name="Övriga Sverige", x=x, y=rest)

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.show()
ts_weekly_deaths = ts_death[death_start:][ts_death['Region Stockholm'] > 0].rolling(7).mean().dropna()



# Create the total score for each participant

totals = [i for i in ts_weekly_deaths['Todays_Total']]



# Create the percentage of the total score the pre_score value for each participant was

pre_rel = [i / j * 100 for i,j in zip(ts_weekly_deaths['Region Stockholm'], totals)]



# Create the percentage of the total score the mid_score value for each participant was

mid_rel = [i / j * 100 for  i,j in zip(ts_weekly_deaths['Västra Götalandsregionen'], totals)]



# Create the percentage of the total score the post_score value for each participant was

post_rel = [i / j * 100 for  i,j in zip(ts_weekly_deaths['Region Skåne'], totals)]



# Create the percentage of the total score the post_score value for each participant was



rest = [i / j * 100 for  i,j in zip(np.sum([ts_weekly_deaths[i] for i in ts_weekly_deaths.columns if i not in ['Region Stockholm', 'Västra Götalandsregionen', 'Region Skåne', 'Todays_Total']], 0), totals)]



import plotly.graph_objects as go

x=ts_weekly_deaths.index



fig = go.Figure(data=[

    go.Bar(name='Region Stockholm', x=x, y=pre_rel),

    go.Bar(name='Västra Götalandsregionen', x=x, y=mid_rel),

    go.Bar(name="Region Skåne", x=x, y=post_rel),

    go.Bar(name="Övriga Sverige", x=x, y=rest)

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.show()
# END OF NOTEBOOK