!pip install pyvis



#package imports

import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd 

import os

import datetime

import cufflinks as cf

import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

import geopandas as gpd

import pyvis

from pyvis.network import Network

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
#load in data from Data NSW

nswdata = pd.read_csv('https://data.nsw.gov.au/data/dataset/aefcde60-3b0c-4bc0-9af1-6fe652944ec2/resource/21304414-1ff1-4243-a5d2-f52778048b29/download/covid-19-cases-by-notification-date-and-postcode-local-health-district-and-local-government-area.csv')

nswages = pd.read_csv('https://data.nsw.gov.au/data/dataset/3dc5dc39-40b4-4ee9-8ec6-2d862a916dcf/resource/24b34cb5-8b01-4008-9d93-d14cf5518aec/download/covid-19-cases-by-notification-date-and-age-range.csv')

nswfactors = pd.read_csv('https://data.nsw.gov.au/data/dataset/c647a815-5eb7-4df6-8c88-f9c537a4f21e/resource/2f1ba0f3-8c21-4a86-acaf-444be4401a6d/download/covid-19-cases-by-notification-date-and-likely-source-of-infection.csv')

ausmapinfo = pd.read_csv('https://raw.githubusercontent.com/matthewproctor/australianpostcodes/master/australian_postcodes.csv')



#amend duplicate postcodes and combine with data

ausmapinfo = ausmapinfo.groupby('postcode').agg({

                             'locality': ' | '.join, 

                             'long':'first',

                             'lat':'first'}).reset_index()

nswmaster = pd.merge(nswdata, ausmapinfo, on='postcode', how='left')
#summarise by cases over time

summarised_cases = nswmaster.groupby('notification_date').agg({'notification_date':'count'}).rename(columns={'notification_date':'Count'}).reset_index()

summarised_cases = summarised_cases.sort_values(by=['notification_date'], ascending=False)

summarised_cases[['notification_date','Count']].head(1).style
#New cases by day in NSW over time

fig = px.bar(summarised_cases, x="notification_date", y="Count", title='New cases by day in NSW over time')

fig.update_layout(xaxis_title="Date", yaxis_title="Count")

fig.show()
summarised_cases['cumulative']=summarised_cases.loc[::-1, 'Count'].cumsum()[::-1]

summarised_cases[['notification_date','cumulative']].head(1).style
fig = px.line(summarised_cases, x="notification_date", y="cumulative", title='Cumulative cases in NSW over time')

fig.update_layout(xaxis_title="Date", yaxis_title="Count")

fig.show()
summarised_cases['Rate of Change']=summarised_cases['cumulative'].pct_change()*-100

summarised_cases = summarised_cases.dropna()

summarised_cases[['notification_date','Rate of Change']].head(1).style
fig = px.line(summarised_cases, x="notification_date", y="Rate of Change", title='Rate of change of new cases by day in NSW over time')

fig.update_layout(xaxis_title="Date", yaxis_title="Rate")

fig.show()
#summarise by suburbs



summarised_suburb = nswmaster.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()

summarised_suburb = summarised_suburb.sort_values(by=['Count'], ascending=False)

summarised_suburb = summarised_suburb.replace('BEN BUCKLER','BONDI/BEN BUCKLER')

summarised_suburb_histogram = summarised_suburb.head(20)



data  = go.Data([

            go.Bar(

              y = summarised_suburb_histogram.Count,

              x = summarised_suburb_histogram.locality,

              orientation='v',

        )])

layout = go.Layout(

        title = "NSW Coronavirus Cases By 20 Most Affected Suburbs"

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
summarised_locals = nswmaster.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()

summarised_locals = summarised_locals.sort_values(by=['Count'], ascending=False)



data  = go.Data([

            go.Bar(

              y = summarised_locals.Count,

              x = summarised_locals.lhd_2010_name,

              orientation='v'

        )])

layout = go.Layout(

        title = "NSW Coronavirus Cases By Region"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
#Prepare Sydney Metro Cases

nswmastermap = nswmaster[nswmaster.long != 0.000000]

nswmastermap = nswmastermap[nswmastermap['postcode'].between(2000, 2234)]

nswmastermap = nswmastermap.groupby('lga_name19').agg({'lga_name19':'count','long':'first','lat':'first'}).rename(columns={'lga_name19':'Count'}).reset_index()
#Map Sydney Metro Cases

fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="lga_name19", hover_data=["Count"],

                        color_discrete_sequence=["red"], zoom=9, height=500, size="Count",size_max=25)

fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
nswmastermap = nswmaster[nswmaster.long != 0.000000]

nswmastermap = nswmastermap[nswmastermap['postcode'].between(2000, 2234)]

nswmastermap = nswmastermap.groupby('locality').agg({'locality':'count','long':'first','lat':'first'}).rename(columns={'locality':'Count'}).reset_index()



#Map Sydney Suburbs

fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="locality", hover_data=["Count"],

                        color_discrete_sequence=["red"], zoom=9, height=500, size="Count",size_max=25)

fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
sydneycases = nswmaster[nswmaster.long != 0.000000]

sydneycases = sydneycases[sydneycases['postcode'].between(2000, 2234)]

sydneycases = sydneycases.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()

sydneycases = sydneycases.sort_values(by=['Count'], ascending=False)



data  = go.Data([

            go.Bar(

              y = sydneycases.Count,

              x = sydneycases.lhd_2010_name,

              orientation='v'

        )])

layout = go.Layout(

        title = "Sydney Cases By Region"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
sydneycasessub = nswmaster[nswmaster.long != 0.000000]

sydneycasessub = sydneycasessub[sydneycasessub['postcode'].between(2000, 2234)]

sydneycasessub = sydneycasessub.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()

sydneycasessub = sydneycasessub.sort_values(by=['Count'], ascending=False)

sydneycasessub = sydneycasessub.replace('BEN BUCKLER','BONDI/BEN BUCKLER')

sydneycasessub_histogram = sydneycasessub.head(20)



data  = go.Data([

            go.Bar(

              y = sydneycasessub_histogram.Count,

              x = sydneycasessub_histogram.locality,

              orientation='v',

        )])

layout = go.Layout(

        title = "Sydney Coronavirus Cases By 20 Most Affected Suburbs"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
#Prepare Non Syd Metro Cases

nswmastermap = nswmaster[nswmaster.long != 0.000000]

nswmastermap = nswmastermap[nswmastermap['postcode'].between(2235, 2999)]

nswmastermap = nswmastermap.groupby('lga_name19').agg({'lga_name19':'count','long':'first','lat':'first'}).rename(columns={'lga_name19':'Count'}).reset_index()
fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="lga_name19", hover_data=["Count"],

                        color_discrete_sequence=["red"], zoom=5, height=500, size="Count",size_max=15)

fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
nonsydneycases = nswmaster[nswmaster.long != 0.000000]

nonsydneycases = nonsydneycases[nonsydneycases['postcode'].between(2235, 2999)]

nonsydneycases = nonsydneycases.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()

nonsydneycases = nonsydneycases.sort_values(by=['Count'], ascending=False)



data  = go.Data([

            go.Bar(

              y = nonsydneycases.Count,

              x = nonsydneycases.lhd_2010_name,

              orientation='v'

        )])

layout = go.Layout(

        title = "Non Sydney Cases By Region"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)

nonsydneycasessub = nswmaster[nswmaster.long != 0.000000]

nonsydneycasessub = nonsydneycasessub[nonsydneycasessub['postcode'].between(2235, 2999)]

nonsydneycasessub = nonsydneycasessub.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()

nonsydneycasessub = nonsydneycasessub.sort_values(by=['Count'], ascending=False)

nonsydneycasessub_histogram = nonsydneycasessub.head(20)



data  = go.Data([

            go.Bar(

              y = nonsydneycasessub_histogram.Count,

              x = nonsydneycasessub_histogram.locality,

              orientation='v',

        )])

layout = go.Layout(

        title = "Non Sydney Coronavirus Cases By 20 Most Affected Suburbs"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
summarised_ages = nswages.groupby('age_group').agg({'age_group':'count'}).rename(columns={'age_group':'Count'}).reset_index()

summarised_ages = summarised_ages.sort_values(by=['Count'], ascending=False)

summarised_ages[['age_group','Count']].head(3).style
data  = go.Data([

            go.Bar(

              y = summarised_ages.Count,

              x = summarised_ages.age_group,

              orientation='v'

        )])

layout = go.Layout(

        title = "Cases by Age Group"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
summarised_factors = nswfactors.groupby('likely_source_of_infection').agg({'likely_source_of_infection':'count'}).rename(columns={'likely_source_of_infection':'Count'}).reset_index()

summarised_factors  = summarised_factors.sort_values(by=['Count'], ascending=False)

summarised_factors[['likely_source_of_infection','Count']].head(3).style
data  = go.Data([

            go.Bar(

              y = summarised_factors.Count,

              x = summarised_factors.likely_source_of_infection,

              orientation='v'

        )])

layout = go.Layout(

        title = "Cases by Likely Source of Infection"

)

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
nswmasterx = nswmaster[['lhd_2010_name','locality']]

nswmasterx['Case ID'] = np.arange(len(nswmasterx))+1

nswmasterx['Case ID'] = nswmasterx['Case ID'].apply(str)

nswmasterx = nswmasterx.dropna()

nswmasterx['Total'] = 1

print(nswmasterx)
casenetwork = Network(height="600px", width="100%", font_color="black", notebook=True)

casenetwork.set_edge_smooth("discrete")



sources = nswmasterx['locality']

targets = nswmasterx['Case ID']



weights = nswmasterx['Total']



edge_data = zip(sources, targets, weights)







for e in edge_data:

    src = e[0]

    dst = e[1]

    w = e[2]

    casenetwork.add_node(src, src, title=src, color="#0000ff", value=10)

    casenetwork.add_node(dst, dst, title=dst, color ="#ff0000", value=2)

    casenetwork.add_edge(src, dst, value=w, color='#00ff00')



#casenetwork.show("covidnswgraph.html")
