# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# color pallette

cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the datasets



df= pd.read_csv('../input/coronavirus-cases-in-india/Covid cases in India.csv')

df_india = df.copy()

df_india.head()
df_time_frame_2 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df_time_frame_2 = df_time_frame_2.loc[:,~df_time_frame_2.columns.str.startswith('Unnamed')]
df_time_frame_2['Total Cases'] = df_time_frame_2['ConfirmedIndianNational']+df_time_frame_2['ConfirmedForeignNational']

df_time_frame_2['Active Cases'] = df_time_frame_2['Total Cases']-(df_time_frame_2['Deaths']+df_time_frame_2['Cured'])
df_time_frame_2.rename(columns={'State/UnionTerritory': 'State'}, inplace=True)
df_time_frame_2.tail(23)
df_time_frame = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')
df_time_frame.tail()
df.drop(['S. No.'],axis=1,inplace=True)
df['Total Cases'] = df['Total Confirmed cases (Indian National)']+df['Total Confirmed cases ( Foreign National )']

df['Active Cases'] = df['Total Cases']-(df['Deaths']+df['Cured/Discharged/Migrated'])
print(f'Total Number CONFIRMED COVID-19 cases across India :',df['Total Cases'].sum())

print(f'Total Number ACTIVE COVID-19 cases across India :',df['Active Cases'].sum())

print(f'Total Number CURED/DISCHARGED/MIGRATED COVID-19 cases across India :',df['Cured/Discharged/Migrated'].sum())

print(f'Total Number DEATHS COVID-19 cases across India :',df['Deaths'].sum())

print(f'Total number of STATES/UTs affected:', df['Name of State / UT'].count())
india_grouped = df.groupby('Name of State / UT')['Total Cases','Active Cases','Cured/Discharged/Migrated','Deaths'].sum().reset_index()
#india_grouped=india_grouped[['Name of State / UT','Total Cases','Active Cases','Cured/Discharged/Migrated','Deaths']]

india_grouped = india_grouped.sort_values(by='Total Cases',ascending=False)

india_grouped = india_grouped.reset_index(drop=True)

india_grouped.style.background_gradient(cmap='Paired')
india_active_cases=india_grouped[['Name of State / UT','Active Cases']]

india_active_cases = india_active_cases.sort_values(by='Active Cases',ascending=False)

india_active_cases = india_active_cases.reset_index(drop=True)

india_active_cases.style.background_gradient(cmap='PuBu')
fig=px.bar(df.sort_values('Active Cases',ascending=False).sort_values('Active Cases',ascending=True),

           x='Active Cases',y='Name of State / UT',title='State with Active Cases'

           , text='Active Cases', orientation='h',

          width=700, height=700, range_x = [0, max(df['Active Cases'])])

fig.update_traces(marker_color='light blue',opacity=0.6)

fig.show()
india_death_cases=india_grouped[['Name of State / UT','Total Cases','Active Cases','Deaths']]

india_death_cases = india_death_cases.sort_values(by='Deaths',ascending=False)

india_death_cases = india_death_cases.reset_index(drop=True)

india_death_cases = india_death_cases[india_death_cases.Deaths>0]

india_death_cases.style.background_gradient(cmap='Reds')
fig=px.bar(india_death_cases.sort_values('Deaths',ascending=False).sort_values('Deaths',ascending=True),

           x='Deaths',y='Name of State / UT',title='State with Deaths Cases',text='Deaths', orientation='h',

          width=700, height=700, range_x = [0, max(india_death_cases['Deaths'])])

fig.update_traces(marker_color='red',opacity=0.6)

fig.show()
india_recovered_cases=india_grouped[['Name of State / UT','Total Cases','Active Cases','Cured/Discharged/Migrated']]

india_recovered_cases = india_recovered_cases.sort_values(by='Cured/Discharged/Migrated',ascending=False)

india_recovered_cases = india_recovered_cases.reset_index(drop=True)

india_recovered_cases = india_recovered_cases[india_recovered_cases['Cured/Discharged/Migrated']>0]

india_recovered_cases.style.background_gradient(cmap='Greens')
fig=px.bar(india_recovered_cases.sort_values('Cured/Discharged/Migrated',ascending=False).sort_values('Cured/Discharged/Migrated',ascending=True),

           x='Cured/Discharged/Migrated',y='Name of State / UT',title='State with Cured/Discharged/Migrated Cases',text='Cured/Discharged/Migrated', orientation='h',

          width=700, height=700, range_x = [0, max(india_recovered_cases['Cured/Discharged/Migrated'])])

fig.update_traces(marker_color='green',opacity=0.6)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_time_frame.Date,y=df_time_frame['Total Cases'],mode='lines+markers',name='Total Cases'))

fig.add_trace(go.Scatter(x=df_time_frame.Date,y=df_time_frame['Active'],mode='lines',name='Active Cases'))

fig.add_trace(go.Scatter(x=df_time_frame.Date,y=df_time_frame['Recovered'],mode='lines',name='Recovered'))

fig.add_trace(go.Scatter(x=df_time_frame.Date,y=df_time_frame['Deaths'],mode='lines',name='Deaths'))

fig.update_layout(title_text='Trend of Coronavirus Cases in India(Cumulative cases)',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
fig=px.bar(df_time_frame,

           x='Date',y='New Cases',title='India with NEW Cases every/day',text='New Cases',

          height=400,barmode='group')

fig.update_traces(marker_color='Red',opacity=0.6)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
date_india = df_time_frame.groupby(['Date'])['Total Cases','New Cases','Deaths', 'Recovered'].sum().reset_index()
temp_spd_india= date_india.melt(id_vars='Date', value_vars=['Total Cases', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_spd_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Cases in INDIA', color_discrete_sequence=cdr)

fig.show()
#'#ff2e63', '#30e3ca

temp_new_india= date_india.melt(id_vars='Date', value_vars=['New Cases','Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_new_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 New Cases/Recovered in INDIA', color_discrete_sequence=['blue','#30e3ca'])

fig.show()
temp_new_india= date_india.melt(id_vars='Date', value_vars=['Recovered','Deaths'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_new_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Recovered/Deaths in INDIA', color_discrete_sequence=['#30e3ca','#ff2e63'])

fig.show()
fig = go.Figure(data=[go.Pie(labels=['Recovered','Deaths','Active'], 

                             values=[df_time_frame['Recovered'].sum(),df_time_frame['Deaths'].sum(),df_time_frame['Active'].sum()] , hole=.3)])



fig.update_traces(hoverinfo='label+percent+value', textinfo='label+percent', textfont_size=18,

                  marker=dict(colors=['mediumturquoise', 'orangered','rainbow'], line=dict(color='#000000', width=2)))



fig.update_layout(

    title_text="Death/Recovered Rate",plot_bgcolor='rgb(250, 242, 242)')



fig.show()
temp_india = df_time_frame.copy()

temp_india['No_of_death_to_100_confirmed_cases'] = round(temp_india['Deaths']/temp_india['Total Cases'],3)*100

temp_india['No_of_Recovered_to_100_confirmed_cases'] = round(temp_india['Recovered']/temp_india['Total Cases'],3)*100

temp_india['No_of_Recovered_to_1_Death_cases'] = round(temp_india['Recovered']/temp_india['Deaths'],3)
temp_india = temp_india.melt(id_vars='Date', 

                 value_vars=['No_of_death_to_100_confirmed_cases', 'No_of_Recovered_to_1_Death_cases','No_of_Recovered_to_100_confirmed_cases'], 

                 var_name='Ratio', 

                 value_name='Value')
fig = px.area(temp_india, x="Date", y="Value", color='Ratio', 

              title='Recovery and Mortality Rate Over The Time',color_discrete_sequence=cdr)

fig.show()
df_time_frame_MH = df_time_frame_2[df_time_frame_2['State']=='Maharashtra']

df_time_frame_KR = df_time_frame_2[df_time_frame_2['State']=='Kerala']

df_time_frame_KN = df_time_frame_2[df_time_frame_2['State']=='Karnataka']

df_time_frame_DL = df_time_frame_2[df_time_frame_2['State']=='Delhi']
import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,start_cell="bottom-left",

    subplot_titles=("Maharashtra","Kerala", "Karnataka","Delhi"))



fig.add_trace(go.Bar(x=df_time_frame_MH['Date'], y=df_time_frame_MH['Total Cases'],

                    marker=dict(color=df_time_frame_MH['Total Cases'], coloraxis="coloraxis")),

              1, 1)



fig.add_trace(go.Bar(x=df_time_frame_KR['Date'], y=df_time_frame_KR['Total Cases'],

                    marker=dict(color=df_time_frame_KR['Total Cases'], coloraxis="coloraxis")),

              1, 2)



fig.add_trace(go.Bar(x=df_time_frame_KN['Date'], y=df_time_frame_KN['Total Cases'],

                    marker=dict(color=df_time_frame_KN['Total Cases'], coloraxis="coloraxis")),

              2, 1)

fig.add_trace(go.Bar(x=df_time_frame_DL['Date'], y=df_time_frame_DL['Total Cases'],

                    marker=dict(color=df_time_frame_DL['Total Cases'], coloraxis="coloraxis")),

              2, 2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Total Confirmed cases(Cumulative)")



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
fig = make_subplots(

    rows=2, cols=2,start_cell="bottom-left",

    subplot_titles=("Maharashtra","Kerala", "Karnataka","Delhi"))



fig.add_trace(go.Scatter(x=df_time_frame_MH['Date'], y=df_time_frame_MH['Total Cases'],

                    marker=dict(color=df_time_frame_MH['Total Cases'], coloraxis="coloraxis")),

              1, 1)



fig.add_trace(go.Scatter(x=df_time_frame_KR['Date'], y=df_time_frame_KR['Total Cases'],

                    marker=dict(color=df_time_frame_KR['Total Cases'], coloraxis="coloraxis")),

              1, 2)



fig.add_trace(go.Scatter(x=df_time_frame_KN['Date'], y=df_time_frame_KN['Total Cases'],

                    marker=dict(color=df_time_frame_KN['Total Cases'], coloraxis="coloraxis")),

              2, 1)

fig.add_trace(go.Scatter(x=df_time_frame_DL['Date'], y=df_time_frame_DL['Total Cases'],

                    marker=dict(color=df_time_frame_DL['Total Cases'], coloraxis="coloraxis")),

              2, 2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Trend of Coronavirus cases")



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
df_time_frame_2.Date = pd.to_datetime(df_time_frame_2['Date'])

df_time_frame_2.Date  = df_time_frame_2['Date'].dt.strftime('%m/%d/%Y')
date_a_r_1 = df_time_frame_2.groupby(['State','Date'])['Total Cases', 'Deaths', 'Cured'].sum().reset_index()
fig = px.bar(date_a_r_1, x="Date", y="Total Cases", color='State', orientation='v', height=600,

             title='Coronavirus Confirmed Cases Over Time INDIA', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
temp_MH_india= df_time_frame_MH.melt(id_vars='Date', value_vars=['Total Cases', 'Deaths', 'Cured'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_MH_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Cases in MAHARASHTRA', color_discrete_sequence=cdr)

fig.show()
temp_KR_india= df_time_frame_KR.melt(id_vars='Date', value_vars=['Total Cases', 'Deaths', 'Cured'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_KR_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Cases in KERALA', color_discrete_sequence=cdr)

fig.show()
temp_DL_india= df_time_frame_DL.melt(id_vars='Date', value_vars=['Total Cases', 'Deaths', 'Cured'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_DL_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Cases in DELHI', color_discrete_sequence=cdr)

fig.show()
temp_KN_india= df_time_frame_KN.melt(id_vars='Date', value_vars=['Total Cases', 'Deaths', 'Cured'],

                var_name='Case', value_name='Count')

fig = px.bar(temp_KN_india, x="Date", y="Count", color='Case', facet_col="Case",

            title='SARS-CoV-2 Cases in KARNATAKA', color_discrete_sequence=cdr)

fig.show()