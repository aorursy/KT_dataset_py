# Import

# ======



# essential libraries

import math

import random  



from datetime import timedelta



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium



# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# for offline ploting

# ===================

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
# Importing the dataset. Checking the names of columns.



df = pd.read_csv('/kaggle/input/covid19-russia-regions-cases/covid19-russia-cases.csv', dtype={'Region_ID':str})

# pop = pd.read_csv('/kaggle/input/covid19-russia-regions-cases/regions-info.csv')

def remove_dot_zeros_from_df(file):

   df=pd.read_csv(file)

   df=df.fillna(0)

   dfcolumnlist=df.columns

   for column in dfcolumnlist:

       try:

           df['Region_ID']=df['Region_ID'].astype(int)

           df['Deaths']=df['Deaths'].astype(int)

       except Exception as e:

           print(e)

   return df    

df=remove_dot_zeros_from_df('/kaggle/input/covid19-russia-regions-cases/covid19-russia-cases.csv')
df.rename(columns = {"Region/City": "Region", "Region/City-Eng": "Region-Eng"}, inplace = True)

df.columns
df.dtypes
df
df.rename(columns = {"Region/City": "Region", "Region/City-Eng": "Region-Eng"}, inplace = True)

df.columns
df=df.fillna(0)
df.drop(df[df['Region'] == 'Diamond Princess'].index, inplace = True) 

df
# df['Region']=df['Region'].astype(int)
df['Deaths']=df['Deaths'].astype(int)
df['Day-Confirmed']=df['Day-Confirmed'].astype(int)
df['Day-Deaths']=df['Day-Deaths'].astype(int)
df['Day-Recovered']=df['Day-Recovered'].astype(int)
df['Confirmed']=df['Confirmed'].astype(int)
df['Recovered']=df['Recovered'].astype(int)
to = (df['Region-Eng'] == 'Tula region')

to = df.loc[to]

to.tail(50)
df.dtypes
# full_table = pd.merge(data,region_info, on=['Region/City'], how='left')

# full_table.duplicated
# Cleaning data

# =============



# Active Case = confirmed - deaths - recovered

to['Active'] = to['Confirmed'] - to['Deaths'] - to['Recovered']



# filling missing values 

# full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)



# fixing datatypes

# full_table['Recovered'] = full_table['Recovered'].astype(int)



# full_table.sample(6)
# Grouped by day, country

# =======================



full_grouped = to.groupby(['Date', 'Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



# new cases ======================================================

temp = full_grouped.groupby(['Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Region'] != temp['Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# renaming columns

temp.columns = ['Region', 'Date', 'Day-Confirmed', 'Day-Deaths', 'Day-Recovered']

# =================================================================



# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Region', 'Date'])



# filling na with 0

full_grouped = full_grouped.fillna(0)



# fixing data types

cols = ['Day-Confirmed', 'Day-Deaths', 'Day-Recovered']

full_grouped[cols] = full_grouped[cols].astype('int')



full_grouped['Day-Confirmed'] = full_grouped['Day-Confirmed'].apply(lambda x: 0 if x<0 else x)



full_grouped.tail()
# Day wise

# ========



# table

day_wise = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'Day-Confirmed'].sum().reset_index()



# number cases per 100 cases

day_wise['Deaths / 100 Cases'] = round((day_wise['Deaths']/day_wise['Confirmed'])*100, 2)

day_wise['Recovered / 100 Cases'] = round((day_wise['Recovered']/day_wise['Confirmed'])*100, 2)

day_wise['Deaths / 100 Recovered'] = round((day_wise['Deaths']/day_wise['Recovered'])*100, 2)



# no. of countries

day_wise['No. of regions'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Region'].unique().apply(len).values



# fillna by 0

cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

day_wise[cols] = day_wise[cols].fillna(0)



day_wise.tail()
# Country wise

# ============



# getting latest values

country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)



# group by country

country_wise = country_wise.groupby('Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'Day-Confirmed'].sum().reset_index()



# per 100 cases

country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)

country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)

country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)



cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

country_wise[cols] = country_wise[cols].fillna(0)



country_wise.head()
pop = pd.read_csv("/kaggle/input/covid19-russia-regions-cases/regions-info.csv")

# pop
pop.dtypes
# load population dataset



# select only population

# pop = pop.iloc[:, :2]



# pop.sample

#rename column names

# pop.columns = ['Region', 'Population']



# merged data

# country_wise = pd.merge(country_wise, pop, on='Region', how='left')



  

# missing values

# country_wise.isna().sum()

# country_wise[country_wise['Population'].isna()]['Region'].tolist()



# Cases per population

# country_wise['Cases / 1k People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000)



# country_wise.head()
temp = to.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=225, width=1200,

                 color_discrete_sequence=[act, rec, dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()
temp = to.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case', height=600,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig_c = px.bar(day_wise, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="Deaths", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Deaths reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)

fig.show()



# ===============================



fig_1 = px.line(day_wise, x="Date", y="Deaths / 100 Cases", color_discrete_sequence = [dth])

fig_2 = px.line(day_wise, x="Date", y="Recovered / 100 Cases", color_discrete_sequence = [rec])

fig_3 = px.line(day_wise, x="Date", y="Deaths / 100 Recovered", color_discrete_sequence = ['#333333'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 

                    subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=1, col=3)



fig.update_layout(height=480)

fig.show()



# ===================================



fig_c = px.bar(day_wise, x="Date", y="Day-Confirmed", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="No. of regions", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('No. of new cases everyday', 'Regions'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)

fig.show()