import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

import matplotlib as mpl

import re

import plotly.graph_objects as go

import plotly.express as px

import random



import matplotlib.pylab as pylab

import folium.plugins

import folium



mpl.style.use('ggplot')
df_confirmed = pd.read_csv('/kaggle/input/corona-virus-report/time_series_2019-ncov-Confirmed.csv')

df_deaths = pd.read_csv('/kaggle/input/corona-virus-report/time_series_2019-ncov-Deaths.csv')

df_recovered = pd.read_csv('/kaggle/input/corona-virus-report/time_series_2019-ncov-Recovered.csv')
df_confirmed.head()
def df_preprocessing(data):

    cn_pr = [] # Create a list of China provinces

    data = data.copy()

    old_columns = df_confirmed.columns.to_list()

    new_columns = old_columns.copy()



    new_columns[0] = 'province'

    new_columns[1] = 'country'

    new_columns[2] = 'lat'

    new_columns[3] = 'long'



    df_confirmed.columns = new_columns

    

    data.columns.values[0] = 'province'

    data.columns.values[1] = 'country'

    data.columns.values[2] = 'lat'

    data.columns.values[3] = 'long'

    

    data.loc[data['country'] == 'Mainland China', 'country'] = 'China'

    

    

    m = data['country'] == 'China'

    cn_pr = data['province'].where(m, 0).to_list()

    cn_pr = list(set([x for x in cn_pr if not isinstance(x, int)]))

    

    

    for i in range(4):

        data.iloc[:,i].fillna('-', inplace=True)

    

    return data, cn_pr
df_conf_tr, cn_conf_provinces = df_preprocessing(df_confirmed)

df_deaths_tr, cn_deaths_provinces = df_preprocessing(df_deaths)

df_recovered_tr, cn_recovered_provinces = df_preprocessing(df_recovered)
df_conf_tr.head()
def day_country_gr(data, p):

    data_gr = data.groupby(p, as_index = False).sum()

    data_gr.drop(['lat', 'long'], inplace=True, axis=1)

    

    return data_gr
c_gr_conf = day_country_gr(df_conf_tr, ['country'])

cp_gr_conf = day_country_gr(df_conf_tr, ['province', 'country'])



c_gr_deaths = day_country_gr(df_deaths_tr, ['country'])

cp_gr_deaths = day_country_gr(df_deaths_tr, ['province', 'country'])



c_gr_recovered = day_country_gr(df_recovered_tr, ['country'])

cp_gr_recovered = day_country_gr(df_recovered_tr, ['province', 'country'])
def plot_stacked(data, data2, name, china = False):

    

    data = data.copy()

    data2 = data2.copy()

       

    data = data.loc[data.iloc[:,1:].sum(axis=1) > 0]

    data2 = data2.loc[data2.iloc[:,1:].sum(axis=1) > 0]

    

    ind = [pd.to_datetime(x) for x in data.columns.to_list()[1:]]

    

    countries = data.country.values



    countries2 = data2.country.values



    provinces = data2.province.values

    

    d = []



    if china == False:

        for i, c in enumerate(countries):

            if c == 'China':

                continue

            d.append(go.Bar(name=c, x=ind, y=data.iloc[i,1:].values))

    else:

        for i, c in enumerate(countries2):

            if c != 'China':

                continue

            d.append(go.Bar(name=data2.iloc[i,0], x=ind, y=data2.iloc[i,2:].values))

    

    plt.figure(figsize=(14,12))

    

    fig = go.Figure(data=d)

    

   # Change the bar mode

    fig.update_layout(barmode='stack')

    if china == True:

        fig.update_layout(title_text = name + ' inside China')

    else:

        fig.update_layout(title_text = name + ' outside China')

    fig.update_layout(xaxis_tickformat = '%b %d')

    fig.update_xaxes(tickangle=90, tickvals=ind)

    

    fig.show()

    

    return data
t1 = plot_stacked(c_gr_conf, cp_gr_conf, 'Confirmed cases', True)
t2 = plot_stacked(c_gr_conf, cp_gr_conf, 'Confirmed cases', False)
t3 = plot_stacked(c_gr_deaths, cp_gr_deaths, 'Deaths', True)
t4 = plot_stacked(c_gr_deaths, cp_gr_deaths, 'Deaths', False)
t5 =  plot_stacked(c_gr_recovered, cp_gr_recovered, 'Recovered', True)
t6 =  plot_stacked(c_gr_recovered, cp_gr_recovered, 'Recovered', False)
def generate_points_df(data):

    points_df = []

    data = data.values.copy()



    for row in range(data.shape[0]):

        if (data[row, 2] > 0) :

            points_df.append(data[row, :2])

        else:

            pass



    points_df = np.array(points_df)

    

    points_df = points_df + np.random.normal(1)/200

    

    return points_df
def plot_heatmap():



    days_list = []



    for day in range(4,df_conf_tr.shape[1]):

        days_list.append(generate_points_df(df_conf_tr.iloc[:,[2,3,day]]).tolist())



    time_index = [str(t.date()) for t in map(pd.to_datetime, df_conf_tr.columns.to_list()[4:])]



    m = folium.Map([30.18, 20.41], zoom_start=2)

    

    folium.TileLayer('cartodbpositron').add_to(m)



    hm = folium.plugins.HeatMapWithTime(days_list,

                                        index=time_index,

                                        name = 'HeatMap',

                                        min_opacity=0.40,

                                        max_opacity=0.8,

                                        overlay=True,

                                        auto_play=True,

                                        control=True,

                                        gradient = {0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1: 'red'},

                                        use_local_extrema=True,

                                        radius=17)



    hm.add_to(m)



    return m
m = plot_heatmap()



m