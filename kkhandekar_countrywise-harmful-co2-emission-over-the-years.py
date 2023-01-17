'''Libraries'''



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



# PLot

import plotly.express as px

import plotly.graph_objects as go

'''Data'''

url = '../input/co2-ghg-emissionsdata/co2_emission.csv'

df = pd.read_csv(url, header='infer')



# Drop Code Column

df.drop('Code', axis=1, inplace=True)



# Rename Emission Column

df.rename(columns={'Annual CO₂ emissions (tonnes )':'Co2_Emissions'}, inplace=True)



# Total Records

print("total records: ", df.shape[0])



# Unique Countries

print("total unique countries: ", df.Entity.nunique())



# Inspect

df.head()
'''Visualisation - Helper Function'''



def visualise(country):

    

    '''Creating a seperate dataframe'''

    df_vis = df[df['Entity'] == country]

    tot_yr = df_vis.Year.max() - df_vis.Year.min()

    tot_em = df_vis.Co2_Emissions.sum()

    print(f"Total Co2 Emissions by {country} in {tot_yr} years: {'{:.2f}'.format(tot_em)} tonnes")

    

    '''Plot'''

    fig = px.line(df_vis, x="Year", y='Co2_Emissions', hover_data={"Co2_Emissions"},

              title='Total Co2 Emissions by '+country+' in '+str(tot_yr)+' years')

    

    fig.show()
# China

visualise('China')
# Brazil

visualise('Brazil')
# Australia

visualise('Australia')
# Hong Kong

visualise('Hong Kong')
# India

visualise('India')
# Japan

visualise('Japan')
# New Zealand

visualise('New Zealand')
# United States

visualise('United States')
# United Kingdom

visualise('United Kingdom')
# Singapore

visualise('Singapore')
# Switzerland

visualise('Switzerland')