# Load packages

import numpy as np 

import pandas as pd

import os



# Big query helpers

from google.cloud import bigquery

from bq_helper import BigQueryHelper



# Import plotting libaries

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.io as pio



# Need this so we can use Plotly in offline mode

# This will allow the maps we make to show up in this notebook

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
# Set up query helpers

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")

pollutants = ['o3','co','no2','so2','pm25_frm']
# This code block based on Air Pollution 101 kernel by Mohammed Jabri

pollutant_year = """

    SELECT

        pollutant.county_name AS county, AVG(pollutant.aqi) AS AvgAQI_pollutant,

        pollutant.state_code, pollutant.county_code

    FROM

      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant

    WHERE

      pollutant.poc = 1

      AND EXTRACT(YEAR FROM pollutant.date_local) = 2016

    GROUP BY 

      pollutant.state_code, pollutant.county_code, pollutant.county_name

"""
# Initialize the data-frame for 2016

df_2016 = pd.DataFrame()



# Now loop through the pollutants list we already have

for elem_g in pollutants : 

    

    # Replaces the word 'pollutant' in the query with the actual pollutant's name

    query = pollutant_year.replace("pollutant", elem_g)

    

    # Runs the query and transforms it to a pandas data-frame 

    # Create a joined up FIPS code that uniquely identifies counties

    # Set the index 

    temp = bq_assistant.query_to_pandas(query)

    temp['location_code'] = temp['state_code'] + temp['county_code']

    temp.set_index('location_code') 

    

    # Concatenate the tables of the different pollutants together 

    # Fill in the missing values with the mean of the column

    if elem_g == 'o3': 

        df_2016 = temp 

    

    # Merge on location code

    else:

        temp.drop(['state_code', 'county_code', 'county'], inplace = True, axis = 1)

        df_2016 = pd.merge(df_2016, temp, how = 'outer', on = ['location_code'],

                          indicator = elem_g + '_merge')



# Randomly pick 10 counties to take a look at the data

df_2016.sample(10,random_state = 42)
# Fill in the numeric missing values 

for column in df_2016.columns: 

    if df_2016[column].dtype in ['float64', 'int64']: 

        df_2016[column].fillna(df_2016[column].mean(), inplace = True)



# Randomly pick 10 counties to take a look at the data

df_2016.sample(10,random_state = 42)
def make_plot(pollutant, plot_labels, color_scale):

    '''This code makes the choloropleth map.'''



    # Store the location codes (also called FIPS codes)

    fips = df_2016['location_code'].tolist()

    values = df_2016['AvgAQI_' + pollutant].tolist()

    

    # Store the end-points 

    endpts = list(np.linspace(min(values), max(values), len(color_scale) - 1))



    # Create the choloropleth map

    fig = ff.create_choropleth(

        fips = fips, values = values, scope = ['usa'],

        binning_endpoints = endpts, colorscale = color_scale,

        show_state_data = False,

        show_hover = True, centroid_marker = {'opacity': 0},

        asp = 2.9, title = 'USA by Average ' + plot_labels[pollutant]['title'],

        legend_title = 'Avg. ' + plot_labels[pollutant]['title']

    )



    # Show the chloropleth map

    iplot(fig, filename = 'choropleth_full_usa')

    

    return
# Run the code

if __name__ == '__main__':



    # Store the labels dictionary 

    plot_labels = {'o3': {'title': 'O3'}, 'co': {'title': 'CO'}, 

                   'pm25_frm': {'title': 'PM 2.5'}, 'no2': {'title': 'NO2'}, 

                  'so2': {'title': 'SO2'}} 



    # Store the color-scale

    color_scale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",

                  "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",

                  "#08519c","#0b4083","#08306b"]

    

    # Make the plot for each pollutant

    for pollutant in pollutants:

        make_plot(pollutant, plot_labels, color_scale)