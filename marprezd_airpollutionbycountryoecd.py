# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the necessary modules or libraries
import plotly.express as px
import plotly.graph_objects as go 
# load our datasets
df_pollution_exposure = pd.read_csv('/kaggle/input/air-pollution-exposure-and-effects/air_pollution_exp_by_country_oecd.csv')
df_pollution_effect = pd.read_csv('/kaggle/input/air-pollution-exposure-and-effects/air_pollution_eff_by_country_oecd.csv')
# Let's look at the air pollution exposure dataframe
df_pollution_exposure
# Let's look at the air pollution effect dataframe
df_pollution_effect
# In df_pollution_exposure dataframe: Drop columns 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes'
df_pollution_exposure = df_pollution_exposure.drop(columns=['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes'])

# In df_pollution_effect: Drop columns 'LOCATION', 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'TIME', 'Flag Codes'
df_pollution_effect = df_pollution_effect.drop(columns=['LOCATION', 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'TIME', 'Flag Codes'])
# Let's look at the air pollution exposure dataframe again
df_pollution_exposure
# Let's look at the air pollution effect dataframe again
df_pollution_effect
# In df_pollution_exposure rename the 'LOCATION', 'TIME', and 'Value' columns
columns_df1 = ['Country', 'Year', 'ExposurePM2.5']
df_pollution_exposure.columns = columns_df1

# In df_pollution_effect rename 'Value' column
columns_df2 = ['MortalityPM2.5']
df_pollution_effect.columns = columns_df2
# Let's join the two data frames
df = pd.concat([df_pollution_exposure, df_pollution_effect], axis=1)
# Let's look at our dataframe again
df
# Round 'ExposurePM2.5' column with 2 decimal and 'MortalityPM2.5' column with 1 decimal
df = df.round({"ExposurePM2.5":2, "MortalityPM2.5":1})
df
# Create a list from 'Year' column and delete duplicate values
years = df['Year'].to_list()
years = list(dict.fromkeys(years))
def air_polluton_exposure(country1, country2):
    """This is a function that accepts two parameters and returns a plot with a group of bars 
    comparing the exposure to air pollution of two countries given by the user."""
    
    # Prepare data frames
    data_country1 = df.loc[df.loc[:, 'Country'] == country1]
    data_country1 = data_country1['ExposurePM2.5'].to_list()
     
    data_country2 = df.loc[df.loc[:, 'Country'] == country2]
    data_country2 = data_country2['ExposurePM2.5'].to_list()
    
    # Initialize the Plotly's plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years,
        y=data_country1,
        name=country1,
        marker_color='olive'
    ))
    fig.add_trace(go.Bar(
        x=years,
        y=data_country2,
        name=country2,
        marker_color='darkolivegreen'
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(
        title='Air Pollution Exposure to PM2.5 fine particles - countries and regions OECD',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Exposure to PM2.5 in Mcg/m3',
            titlefont_size=16,
            tickfont_size=14,
            tickformat=".2f"
        ),
        barmode='group', xaxis_tickangle=-45)
    
    # Return the plot
    return fig.show()
# Receive user input
first_country = input('Enter first country to compare: ')
second_country = input('Enter second country to compare: ')

# Call to 'air_polluton_exposure(params1, params2)' function
plot = air_polluton_exposure(first_country.title(), second_country.title())
plot
# Initialize the while loop...
update_plot = True
while update_plot:
    question = int(input('Do you want to compare another group of countries? (Enter 1 or 0).' +
                        '\n1 = continue, 0 = exit: '))
    if question == 1:
        previous_countries = input('Do you want to compare ' + str(first_country ).upper() + ' or ' + 
                             str(second_country).upper() + ' with another country? (Enter f, s or n).'
                        '\nf = choose the first country, s = choose the second country, n = neither: ')
        if previous_countries == 'f':
            first_country = first_country
            print('You chose:', first_country.title())
            second_country = input('Enter another country to compare: ')
        elif previous_countries == 's':
            first_country = second_country
            print('You chose:', first_country.title())
            second_country = input('Enter another country to compare: ')
        elif previous_countries == 'n':
            first_country = input('Enter first country to compare: ')
            second_country = input('Enter second country to compare: ')
        
        up_plot = air_polluton_exposure(first_country.title(), second_country.title())
        up_plot

    else:
        print('Thanks! Run this cell again to compare other countries...')
        update_plot = False