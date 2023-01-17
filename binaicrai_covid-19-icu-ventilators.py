# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.graph_objs as gobj

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



import plotly.express as px       

import plotly.offline as py       

import plotly.graph_objects as go 

from plotly.subplots import make_subplots



import re
pd.set_option('display.max_columns', 200)

us_hosp_cap = pd.read_csv('/kaggle/input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-60-population-contracted.csv')

us_death_cases = pd.read_csv('/kaggle/input/uncover/UNCOVER/USAFacts/confirmed-covid-19-deaths-in-us-by-state-and-county.csv')

us_confirmed_cases = pd.read_csv('/kaggle/input/uncover/UNCOVER/USAFacts/confirmed-covid-19-cases-in-us-by-state-and-county.csv')

display(us_hosp_cap.head())

display(us_death_cases.head())

display(us_confirmed_cases.head())
deaths_by_states = us_death_cases.groupby(['state_name'])['deaths'].sum().to_frame(name = 'sum')

deaths_by_states = deaths_by_states.sort_values('sum', ascending = False).reset_index()

deaths_by_states.head()
confirmed_by_states = us_confirmed_cases.groupby(['state_name'])['confirmed'].sum().to_frame(name = 'sum')

confirmed_by_states = confirmed_by_states.sort_values('sum', ascending = False).reset_index()

confirmed_by_states.head()
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=deaths_by_states['state_name'], y=deaths_by_states['sum'], name='<b>Deaths by State', marker_color='rgb(255, 64, 64)'), secondary_y = True)

fig.add_trace(go.Bar(x=confirmed_by_states['state_name'],y=confirmed_by_states['sum'],name='<b>Confirmed by State',marker_color='rgb(255, 185, 15)'))



config = dict({'scrollZoom': True})



fig.update_layout(title='<b>Analysis of Death and Confirmed Cases',xaxis_tickfont_size=14,

                  yaxis=dict(title='<b>Confirmed',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)



fig.update_xaxes(title_text="<b>State")

fig.update_yaxes(title_text="<b>Deaths", secondary_y=True)



fig.show()
combined_df = pd.merge(deaths_by_states, confirmed_by_states, on = 'state_name')

combined_df = combined_df.rename(columns = {'sum_x': 'deaths', 'sum_y': 'confirmed'})

combined_df['death_ratio'] = (combined_df['deaths']/combined_df['confirmed'])*100

combined_df = combined_df.sort_values('death_ratio', ascending = True).reset_index(drop=True).round(3)

combined_df.head()
plt.figure(figsize=(20,20))

plt.barh(combined_df['state_name'], combined_df['death_ratio'])

plt.title('Death Ratio by State wr.r.t Confirmed Cases', fontsize = 20, color = 'black')

for index, value in enumerate(combined_df['death_ratio']):

    plt.text(value, index, str(value))
us_hosp_cap.head()
# Extracting Columns for ICU

import re

l = ['state', 'projected_infected_individuals']

col = us_hosp_cap.columns.to_list()



for i in col:

        # The . symbol is used in place of ? symbol

        if re.search('icu', i) : 

                l.append(i)

l
icu_df = us_hosp_cap[l]

icu_df.head()
icu_df_1 = icu_df[['state', 'potentially_available_icu_beds', 'projected_individuals_needing_icu_care']]

icu_df_1['ratio'] = (icu_df_1['projected_individuals_needing_icu_care']/icu_df_1['potentially_available_icu_beds'])*100

icu_df_1 = icu_df_1.sort_values('ratio', ascending=False).reset_index(drop=True).round()

icu_df_1.head()
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=icu_df_1['state'], y=icu_df_1['potentially_available_icu_beds'], name='Potentially Available ICU Beds', marker_color='rgb(162, 205, 90)'), secondary_y = True)

fig.add_trace(go.Bar(x=icu_df_1['state'],y=icu_df_1['projected_individuals_needing_icu_care'],name='Projected Individuals Needing ICU Care',marker_color='rgb(255, 185, 15)'))

fig.add_trace(go.Bar(x=icu_df_1['state'], y=icu_df_1['ratio'], name='Ratio', marker_color='rgb(255, 64, 64)'))



fig.update_layout(title='<b>Potentially Available ICU Beds & Projected Individuals Needing ICU Care',xaxis_tickfont_size=14,

                  yaxis=dict(title='<b>Project Individuals Needing ICU Care Count',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)





fig.update_xaxes(title_text="<b>State")

fig.update_yaxes(title_text="<b>Potentially Available ICU Beds Count", secondary_y=True)



fig.show()
icu_df_2 = icu_df[['state', 'projected_infected_individuals', 'projected_individuals_needing_icu_care']]

icu_df_2['ratio'] = (icu_df_2['projected_infected_individuals']/icu_df_2['projected_individuals_needing_icu_care'])

icu_df_2 = icu_df_2.sort_values('ratio', ascending=False).reset_index(drop=True).round(2)

icu_df_2.head()
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=icu_df_2['state'], y=icu_df_2['projected_infected_individuals'], name='Projected Infected Individuals', marker_color='rgb(162, 205, 90)'))

fig.add_trace(go.Bar(x=icu_df_2['state'],y=icu_df_2['projected_individuals_needing_icu_care'],name='Projected Individuals Needing ICU Care',marker_color='rgb(255, 185, 15)'))

fig.add_trace(go.Scatter(x=icu_df_2['state'], y=icu_df_2['ratio'], name='Ratio', marker_color='rgb(255, 64, 64)'), secondary_y = True)



fig.update_layout(title='<b>Projected Infected Individuals & Projected Individuals Needing ICU Care',xaxis_tickfont_size=14,

                  yaxis=dict(title='<b>Project Individuals Needing ICU Care Count',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)





fig.update_xaxes(title_text="<b>State")

fig.update_yaxes(title_text="<b>Ratio", secondary_y=True)



fig.show()