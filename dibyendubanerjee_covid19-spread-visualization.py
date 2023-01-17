import plotly.express as px



import pandas as pd

import plotly.graph_objects as go

import numpy as np 



url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'



def read_file(url):

    df = pd.read_csv(url)

    return df



def filter_specific_country(df, selected_countries):

    df1=df[df['Country/Region'].isin(selected_countries) ]

    countrywise_grouped_df = df1.groupby(df['Country/Region']).sum().drop(['Lat','Long'], axis=1)

    countrywise_grouped_df

    return countrywise_grouped_df



def transpose_and_reformat_data(df):

    df_t=df.transpose()

    df_t.reset_index(inplace=True)

    df_t.rename(columns={'Country/Region':'Index_Col', 'index':'Dates'}, inplace=True)

    return df_t



confirmed_dataset = read_file(url)

selected_countries=['India','China','Italy','Spain','France','Australia','Germany','Japan','Korea, South','Pakistan',

                    'Russia','United Kingdom','Canada','Iran','Brazil','Singapore','South Africa','US']

ds=filter_specific_country(confirmed_dataset,selected_countries)

data=transpose_and_reformat_data(ds).melt(id_vars=["Dates"], var_name="Country", value_name="Confirmed_Count")

#plot_title="Global Spread of Covid-19 : (Selected Countries)"

plot_title='Visualizing the spread of Novel Coronavirus COVID-19 (2019-nCoV) - Created by Dibyendu Banerjee'

fig = px.bar(data, y="Country", x="Confirmed_Count", color="Country",

  animation_frame="Dates", range_x=[1,1700000], orientation='h' )

fig.update_layout(title=plot_title,yaxis_title='Countries', xaxis_tickangle=90, font=dict(family="Arial",size=10,color="#7f7f7f"))

fig.show()