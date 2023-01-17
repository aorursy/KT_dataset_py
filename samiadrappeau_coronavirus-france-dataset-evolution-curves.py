from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import plotly.express as px
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def dateFirstCase(data):

    _tmp = pd.DataFrame(data.groupby('region').apply(lambda dr: dr[dr['evolution_cumcount']>0]['confirmed_date'].min())).reset_index().rename(columns={0:'date'})

    _dict = {}

    for row in _tmp.itertuples():

        _dict[row[1]] = row[2]

    return _dict
def growthRate(data):

    pass
def evolutionRegion(data):

    # TODO -- Improve on hardcoding 'France' in columns mapping because we only have one country.

    res = data.groupby(['confirmed_date', 'region'])['country'].value_counts().unstack().rename(columns={'France': 'evolution_count'}).reset_index() 

    full_data = []

    for date, _res in res.groupby(['confirmed_date']):

        missing_regions = list(set(lRegions)- set(_res['region'].tolist()))

        for region in missing_regions:

            dd = {'region': region, 'evolution_count': 0, 'confirmed_date': date}

            _res = _res.append(dd, True)

        full_data.append(_res)

    res_full = pd.concat(full_data)

    # TODO -- Somehow, `res` has a name equal to 'country'. Can't get rid of that

    res_full['date_of_confirmed_infection'] = res_full['confirmed_date'].apply(lambda row: row.strftime("%d-%b-%Y"))

    res_full['evolution_cumcount'] = res_full.groupby('region')['evolution_count'].transform(pd.Series.cumsum)

    

    map_first_day_region = dateFirstCase(res_full)

    

    res_full['date_of_first_confirmed_infection'] = res_full['region'].map(map_first_day_region) 



    return res_full
def plotEvolutionCurve(evolution_data):

    region_names = [i for i in evolution_data.region.unique()]

    region_names.sort()

    max_pop = evolution_data['evolution_cumcount'].max() + 25



    fig = px.scatter(evolution_data, x='date_of_confirmed_infection', y='evolution_cumcount', color='region',

                     category_orders={'region': region_names},

                     opacity=0.7,

                     hover_name='region',hover_data=[],

                     title='Evolution of the number of person infected, per region',

                     labels={'evolution_cumcount': 'Nb of confirmed cases, to date', 'date_of_confirmed_infection': 'Date of confirmed infection'},

                     range_y=[-1,max_pop],

                     template='plotly_white')

    fig.update_traces(mode='lines')

    fig.show()
def plotInfectionCurve(evolution_data):

    region_names = [i for i in evolution_data.region.unique()]

    region_names.sort()

    max_pop = evolution_data['evolution_cumcount'].max() + 25



    fig = px.scatter(evolution_data, x='Days_since_1st_case', y='evolution_cumcount', color='region',

                     category_orders={'region': region_names},

                     opacity=0.7,

                     #hover_name='region',hover_data=[],

                     title='Infection curve, per region',

                     labels={'evolution_cumcount': 'Nb of confirmed cases', 'Days_since_1st_case': 'Days since 1st confirmed infection'},

                     range_y=[-1,max_pop],

                     template='plotly_white',

                     trendline="lowess")

    #fig.update_traces(mode='markers+lines')

    fig.update_traces(

     line=dict(dash="dot", width=2),

     selector=dict(type="scatter", mode="lines"))

    fig.show()
def plotTimelapseRegion(data):

    region_names = [i for i in data.region.unique()]

    region_names.sort()

    max_pop = data['evolution_cumcount'].max() + 25

    max_date = data['date_of_confirmed_infection'].max()

    min_date = data['date_of_confirmed_infection'].min()

    

    fig = px.scatter(data, x='evolution_cumcount', y='date_of_first_confirmed_infection', color='region',

                     category_orders={'region': region_names},

                     hover_name='region',hover_data=[],

                     text='region',

                     animation_frame='date_of_confirmed_infection', animation_group='region',

                     title='Timelapse of Coronavirus Cases by Region (ordered by date of first registered case)',

                     labels={'evolution_cumcount': 'Nb of confirmed cases', 'date_of_first_confirmed_infection': 'Date of first confirmed infection'},

                     range_x=[-1, max_pop], range_y=[max_date, min_date],

                     template='plotly_white')

    fig.update_traces(textposition='top center')

    fig.show()
# Loading data

df = pd.read_csv('/kaggle/input/patient.csv', delimiter=',')

df.dataframeName = 'patient.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns.')
nRegions = df['region'].nunique()

lRegions = sorted(df['region'].unique().tolist())

print(f'Currently, there are {nRegions} regions in the dataset:\n{lRegions}.')
df.head(5)
# Cleaning Date columns

df['confirmed_date'] = pd.to_datetime(df['confirmed_date'])
df_evolution = evolutionRegion(df)



df_evolution.reset_index(inplace=True)

del df_evolution['index']
def days_since(df_sub):

    df_sub['Days_since_1st_case'] = df_sub['confirmed_date'] - df_sub['date_of_first_confirmed_infection']

    df_sub['Days_since_1st_case'] = df_sub['Days_since_1st_case'].dt.days

    return df_sub



df_evolution = df_evolution.groupby('region').apply(days_since)
plotEvolutionCurve(df_evolution)
plotTimelapseRegion(df_evolution)
plotInfectionCurve(df_evolution)