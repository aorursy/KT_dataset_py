import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.options.mode.chained_assignment = None

import datetime



#covid_19_df = pd.read_excel("/kaggle/input/covid19geographicdistributionworldwide/COVID-19-geographic-disbtribution-worldwide.xls",sheet_name="CSV_4_COMS")

#covid_19_df = pd.read_csv("/kaggle/input/covid19geographicdistributionworldwide/COVID-19-geographic-disbtribution-worldwide.csv",encoding = "ISO-8859-1")

covid_19_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid_19_df.columns = ['SNo','DateRep','CountryState', 'CountryExp','UpdateDate' ,'NewConfCasesCum', 'NewDeathsCum', 'NewRecoveredCum']

covid_19_df = covid_19_df[covid_19_df['CountryState'].isnull()]

covid_19_df['DateRep'] = pd.to_datetime(covid_19_df['DateRep'])

population_df = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")

# Any results you write to the current directory are saved as output.
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus'])



covid_19_df_eu = covid_19_df[covid_19_df['CountryExp'].isin(europe)]

population_df_eu = population_df[population_df['Country (or dependency)'].isin(europe)]
#covid_19_df = covid_19_df[covid_19_df["EU"] == 'EU'].reset_index()

'''

cumulative_df = covid_19_df_eu.sort_values(["CountryExp",'DateRep'],ascending=True).reset_index()



cumulative_df1 = cumulative_df.groupby(["CountryExp"])["NewConfCases"].cumsum().reset_index()

cumulative_df1['NewConfCasesCum'] = cumulative_df1['NewConfCases']

cumulative_df1 = cumulative_df1.drop(['NewConfCases'], axis = 1)



cumulative_df2 = cumulative_df.groupby(["CountryExp"])["NewDeaths"].cumsum().reset_index()

cumulative_df2['NewDeathsCum'] = cumulative_df2['NewDeaths']

cumulative_df2 = cumulative_df2.drop(['NewDeaths'], axis = 1)



result = pd.concat([cumulative_df,cumulative_df1,cumulative_df2], axis=1)

'''


result = covid_19_df.sort_values(by="DateRep").reset_index(drop=True)



start_date = datetime.date(2020, 2, 24)

result = result[result["DateRep"]>=start_date]

result["DateRep"] = result["DateRep"].astype(str)





fig = px.choropleth(locations=result['CountryExp'],

                    color=result['NewConfCasesCum'], 

                    locationmode="country names",

                    scope="europe",

                    animation_frame=result["DateRep"],

                    color_continuous_scale='Rainbow',

                    range_color=[0,25000]

                    #autocolorscale=False,

                   )



layout = go.Layout(

    title=go.layout.Title(

        text="Cumulative count of COVID-19 cases in Europe",

        x=0.5

    ),

    font=dict(size=14),

)



fig.update_layout(layout)

fig.show()
#covid_19_pop = pd.merge(covid_19_df_eu, population_df_eu, how='left', left_on='Countries and territories', right_on='Country (or dependency)')



covid_19_df_date = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid_19_df_date.columns = ['SNo','DateRep','CountryState', 'CountryExp','UpdateDate' ,'NewConfCasesCum', 'NewDeathsCum', 'NewRecoveredCum']





covid_19_df_date.index = pd.to_datetime(covid_19_df_date['DateRep'])

covid_19_df_eu = covid_19_df_date[covid_19_df_date['CountryExp'].isin(europe)]

covid_19_df_eu = covid_19_df_eu[covid_19_df_eu['CountryState'].isnull()]



population_df_eu = population_df[population_df['Country (or dependency)'].isin(europe)]



'''

cumulative_df = covid_19_df_eu.sort_values(["CountryExp"],ascending=True)

cumulative_df = cumulative_df.sort_index()



cumulative_df['CaseCum'] = cumulative_df.groupby(["CountryExp"])["NewConfCases"].cumsum()

cumulative_df['DeathsCum'] = cumulative_df.groupby(["CountryExp"])["NewDeaths"].cumsum()



#cumulative_df = cumulative_df[(cumulative_df.index > '2020-03-01') & (cumulative_df.index < '2020-03-31')]

'''
j = 1

data1 = []

for country in covid_19_df_eu["CountryExp"].unique() :

    country_df = covid_19_df_eu[covid_19_df_eu['CountryExp'] == country]

    #country_df['CaseCum'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)    

    Traces = go.Scatter(

         x = country_df.index,

         y = country_df['NewConfCasesCum'],

         mode = 'lines',

         name = country

     )

    data1.append(Traces) 

    j = j + 1



j = 1

data2 = []

for country in covid_19_df_eu["CountryExp"].unique() :

    country_df = covid_19_df_eu[covid_19_df_eu['CountryExp'] == country]

    #country_df['CaseCum'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)    

    Traces = go.Scatter(

         x = country_df.index,

         y = country_df['NewDeathsCum'],

         mode = 'lines',

         name = country

     )

    data2.append(Traces) 

    j = j + 1

layout = go.Layout(

      xaxis=dict(title='Date'),

      yaxis=dict(title='No of Corona Patients (Cumulative figure)'),

      title=('No of Corona Patients across Europe(Cumulative figure)'))

fig = go.Figure(data=data1, layout=layout)

fig
layout = go.Layout(

      xaxis=dict(title='Date'),

      yaxis=dict(title='No of Deaths (Cumulative figure)'),

      title=('No of Deaths reported across Europe(Cumulative figure)'))

fig = go.Figure(data=data2, layout=layout)

fig
#cumulative_df['DateIndex'] = cumulative_df.index

covid_19_df_eu['DateIndex'] = covid_19_df_eu.index



covid_19_eu_pop = pd.merge(covid_19_df_eu, population_df_eu, how='left', left_on='CountryExp', right_on='Country (or dependency)',left_index = True)

covid_19_eu_pop = covid_19_eu_pop[covid_19_eu_pop['DateIndex'] > '2020-02-22']
covid_19_eu_pop['RiskWeight_Cases'] = (covid_19_eu_pop['NewConfCasesCum'] / covid_19_eu_pop['Population (2020)']) * 1000

covid_19_eu_pop['RiskWeight_Deaths'] =(covid_19_eu_pop['NewDeathsCum']/ covid_19_eu_pop['Population (2020)']) * 100000
j = 1

data3 = []

for country in covid_19_eu_pop["CountryExp"].unique() :

    country_df = covid_19_eu_pop[covid_19_eu_pop['CountryExp'] == country]

    #country_df['CaseCum'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)    

    Traces = go.Scatter(

         x = country_df['DateIndex'],

         y = country_df['RiskWeight_Cases'],

         mode = 'lines',

         name = country

     )

    data3.append(Traces) 

    j = j + 1

    

j = 1

data4 = []

for country in covid_19_eu_pop["CountryExp"].unique() :

    country_df = covid_19_eu_pop[covid_19_eu_pop['CountryExp'] == country]

    #country_df['CaseCum'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)    

    Traces = go.Scatter(

         x = country_df['DateIndex'],

         y = country_df['RiskWeight_Deaths'],

         mode = 'lines',

         name = country

     )

    data4.append(Traces) 

    j = j + 1
layout = go.Layout(

      xaxis=dict(title='Date'),

      yaxis=dict(title='(Patient reported/Population)*1000'),

      title=('Which country is at risk, Ranking => (Patient reported/Population)*1000 '))

fig = go.Figure(data=data3, layout=layout)

fig
index_eu = covid_19_eu_pop.groupby(['CountryExp'], sort=False)['RiskWeight_Cases'].max().index

index_values = covid_19_eu_pop.groupby(['CountryExp'], sort=False)['RiskWeight_Cases'].max().values

data = {'weights':index_values} 

df_eu_tot = pd.DataFrame(data, index = index_eu) 

df_eu_tot['CountryExp'] = df_eu_tot.index

df_eu_tot = df_eu_tot.dropna()



fig = px.bar(df_eu_tot.sort_values('weights', ascending=False).sort_values('weights', ascending=True), 

             x="weights", y="CountryExp", 

             title='Cases reported per population of country', 

             text='weights', 

             orientation='h', 

             width=1000, height=700, range_x = [0, max(df_eu_tot['weights'])])

fig.update_traces(marker_color='#46cdfb', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
layout = go.Layout(

      xaxis=dict(title='Date'),

      yaxis=dict(title='(Death reported/Population)*100000'),

      title=('Which country is at risk, Ranking => (Death reported/Population)*100000 '))

fig = go.Figure(data=data4, layout=layout)

fig
index_eu = covid_19_eu_pop.groupby(['CountryExp'], sort=False)['RiskWeight_Deaths'].max().index

index_values = covid_19_eu_pop.groupby(['CountryExp'], sort=False)['RiskWeight_Deaths'].max().values

data = {'weights':index_values} 

df_eu_tot = pd.DataFrame(data, index = index_eu) 

df_eu_tot['CountryExp'] = df_eu_tot.index

df_eu_tot = df_eu_tot.dropna()



fig = px.bar(df_eu_tot.sort_values('weights', ascending=False).sort_values('weights', ascending=True), 

             x="weights", y="CountryExp", 

             title='Deaths reported per population of country', 

             text='weights', 

             orientation='h', 

             width=1000, height=700, range_x = [0, max(df_eu_tot['weights'])])

fig.update_traces(marker_color='#46cdfb', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()