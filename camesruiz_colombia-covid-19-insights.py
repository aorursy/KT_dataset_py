# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

from matplotlib.pyplot import plot

import seaborn as sn

%matplotlib inline 



import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex3 import *



from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
#Data import



colombia_df = pd.read_csv('../input/colombia-covid19-complete-dataset/covid-19-colombia-all.csv')

confirmed = pd.read_csv('../input/colombia-covid19-complete-dataset/covid-19-colombia-confirmed.csv')

deaths = pd.read_csv('../input/colombia-covid19-complete-dataset/covid-19-colombia-deaths.csv',encoding='ISO-8859-1')

col_df = pd.read_csv('../input/colombia-covid19-complete-dataset/covid-19-colombia.csv')

cases = pd.read_csv('../input/colombia-covid19-complete-dataset/Casos1.csv')
cases.columns = ["ID", "date", "city", "departamento", "state", "age", "sex", "type", "procedence"]

cols_ = cases.select_dtypes(include=[np.object]).columns

cases[cols_] = cases[cols_].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))



cases.date = pd.to_datetime(cases.date, dayfirst=True)



cases
cases.groupby(['departamento']).count()
#Dataframe overview

col_df.head()
#Number of actual active cases calculation

col_df['active'] = col_df.confirmed - col_df.deaths - col_df.recovered
def plot_line(df):

    fig.add_shape(

       # Line Horizontal

          type="line",

            x0='2020-03-25',

            y0=0,

            x1='2020-03-25',

            y1=df.max(),

            line=dict(

                color="LightSeaGreen",

                width=4,

                

            ))

    fig.update_layout(

        showlegend=False,

        annotations=[

            dict(

                x='2020-03-25',

                y=df.max(),

                xref="x",

                yref="y",

                text="Quarantine Start",

                )

        ]

    )

    fig.show()
#Plotting

fig = px.line(col_df, x="date", y="confirmed", 

              title="Colombia Confirmed Cases")

plot_line(col_df.confirmed)





fig = px.line(col_df, x="date", y="deaths", 

              title="Colombia Confirmed Deaths")

plot_line(col_df.deaths)



fig = px.line(col_df, x="date", y="recovered", 

              title="Colombia Confirmed Recoveries")

plot_line(col_df.recovered)
fig = px.line(col_df, x="date", y="active", 

              title="Colombia Active Cases")

plot_line(col_df.active)
cols = confirmed.keys()

cols_d = deaths.keys()



confirmed1 = confirmed.loc[:, cols[1]:cols[-1]]

deaths1 = deaths.loc[:, cols_d[1]:cols_d[-1]]
#Number of days since the outbreak (March 6th)

days = np.array([i for i in range(len(col_df.index))]).reshape(-1, 1)



dates = confirmed1.keys()

state_cases = []

total_deaths = [] 



#Total number of cases

for i in dates:

    confirmed_sum = confirmed1[i].sum()

    #death_sum = deaths1[i].sum()

    

    state_cases.append(confirmed_sum)

    #total_deaths.append(death_sum)



print('Total number of confirmed cases: ',confirmed_sum)
state_cases = np.array(state_cases).reshape(-1, 1)

#total_deaths = np.array(total_deaths).reshape(1, -1)



deptos = np.array(confirmed.state)

total = np.array(confirmed.loc[:,cols[-1]])

#total_d = np.array(deaths.loc[:,cols[-1]])
#Loading shapefile for the choropleth

deptos_geo = gpd.read_file('../input/colombia-shape-files-by-departments/depto.shp')

deptos_geo['NOMBRE_DPT'] = deptos_geo['NOMBRE_DPT'].str.capitalize()

deptos_geo.loc[2,'NOMBRE_DPT'] = 'Bogota'

deptos_geo.loc[32,'NOMBRE_DPT'] = 'San andres y providencia'

deptos_geo.loc[16,'NOMBRE_DPT'] = 'Narino'

deptos_geo = deptos_geo.sort_values(by=['NOMBRE_DPT'])

deptos_geo = deptos_geo[['NOMBRE_DPT','geometry']]

deptos_geo.set_index('NOMBRE_DPT', inplace=True)
#List of confirmed cases per state

df = pd.DataFrame({'NOMBRE_DPT':deptos,'confirmed':total})

df['NOMBRE_DPT'] = df['NOMBRE_DPT'].str.capitalize()

df = df.sort_values(by=['NOMBRE_DPT'])

df.set_index('NOMBRE_DPT', inplace=True)
df_merge = deptos_geo.merge(df,on='NOMBRE_DPT') 

df_merge
m_1 = folium.Map(location=[4,-73], tiles='cartodbpositron', zoom_start=5)



folium.Choropleth(geo_data=df_merge['geometry'],

           data=df_merge, columns=[df_merge.index, 'confirmed'],

           key_on="feature.id",

           fill_color='YlOrRd',

           legend_name='Number of confirmed cases'

           ).add_to(m_1)



embed_map(m_1, 'q_1.html')
#Plotting

df = pd.DataFrame({'state':deptos,'confirmed':total})

fig = px.bar(df.sort_values('confirmed', ascending=False)[:10][::-1], 

             x='confirmed', y='state', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Cases by State', text='confirmed', orientation='h')

fig.show()
total_deaths = []

total_deaths = cases.loc[cases['state'] == 'Fallecido']

total_deaths = total_deaths.groupby(['departamento']).count()

total_deaths = total_deaths.rename(columns={"ID": "n"}).drop(columns=['date','city','state','age', 'sex','type','procedence'])



total_deaths
df = pd.DataFrame({'state':total_deaths.index,'confirmed':total_deaths.n})

fig = px.bar(df.sort_values('confirmed', ascending=False)[:10][::-1], 

             x='confirmed', y='state', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Deaths by State', text='confirmed', orientation='h')

fig.show()
#Mortality and recovery rate calculation

col_df['death_rate'] = (col_df.deaths/col_df.confirmed) * 100

col_df['recover_rate'] = (col_df.recovered/col_df.confirmed) * 100

col_df['inf_rate'] = (col_df.confirmed/48258494) * 100



col_df
#Temp dataframe for plotting multiple traces

df = pd.DataFrame([col_df.date,col_df.death_rate,col_df.recover_rate])

df_melt = col_df.melt(id_vars='date', value_vars=['death_rate', 'recover_rate'])
fig = px.line(df_melt, x="date", y="value", 

              title="Colombia Mortality and Recover Rate (%)",color='variable')



print('Death Rate: ',col_df.death_rate.max() , '%')

print('Recover Rate: ',col_df.recover_rate.max() , '%')

plot_line(df_melt.value)
fig = px.line(col_df, x="date", y="inf_rate", 

              title="Colombia Infection Rate (%) (Population: 48'258.494)")



print('Infecion Rate: ',col_df.inf_rate.max() , '%')

plot_line(col_df.inf_rate)
df = pd.DataFrame({'Date':col_df.date,'Confirmed':col_df.confirmed_daily})

fig = px.bar(df, y='Confirmed', x='Date', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Daily Cases', text='Confirmed', orientation='v')

fig.show()



df = pd.DataFrame({'Date':col_df.date,'Deaths':col_df.deaths_daily})

fig = px.bar(df, y='Deaths', x='Date', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Daily Deaths', text='Deaths', orientation='v')

fig.show()



df = pd.DataFrame({'Date':col_df.date,'Recovered':col_df.recovered_daily})

fig = px.bar(df, y='Recovered', x='Date', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Daily Recoveries', text='Recovered', orientation='v')

fig.show()
df_melt = col_df.melt(id_vars='date', value_vars=['recovered_daily','deaths_daily', 'confirmed_daily'])

fig = px.bar(df_melt, y='value', x='date', color='variable',

             title='Confirmed Daily Cases', text='value', orientation='v',barmode='group')

fig.show()
male = cases.loc[cases['sex'] == 'M'].count()[0]

female = cases.loc[cases['sex'] == 'F'].count()[0]



sex_grouped = pd.DataFrame({'M': [male], 'F': [female]}).T

sex_grouped.columns = ['n']

sex_grouped
fig = px.pie(sex_grouped, values='n', names= sex_grouped.index,

             title='Cases by Sex')

fig.show()
age_grouped = cases.groupby(['age']).count()

age_grouped['ID']
age_grouped
#fig = px.pie(age_grouped, values='ID', names= age_grouped.index,

#             title='Cases by Age Groups')

fig = px.bar(age_grouped, y='ID', x=age_grouped.index,

             title='Confirmed Daily Cases', text=age_grouped.index, orientation='v',barmode='group')

fig.show()
state = cases.groupby(['state']).count()

state = state.rename(columns={"ID": "n"}).drop(columns=['date','city','departamento','age', 'sex','type','procedence'])



state
fig = px.bar(state, y='n', x= state.index,

             title='Cases Actual State')

fig.show()
uci = cases.loc[cases['state'] == 'Hospital UCI'].groupby(['date','state']).count().reset_index()
dates = uci.date

total_uci = [uci.iloc[0,2]]



#total_uci.append(uci.iloc[0,2])

for i in range(len(dates)-1):

    uci_sum = total_uci[i] + uci.iloc[i+1,2]

    total_uci.append(uci_sum)

    

   



df = pd.DataFrame({'date':dates,'total':total_uci})
fig = px.line(df, x="date", y="total", 

              title="Actual Active Cases in ICU")

#fig.add_shape(

#       # Line Horizontal

#          type="line",

#            x0='2020-03-16',

#            y0=5359,

#            x1='2020-03-27',

#            y1=5359,

#            line=dict(

#                color="LightSeaGreen",

#                width=4,

#                dash="dashdot",

#            ))



plot_line(df.total)