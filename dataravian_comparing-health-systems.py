import numpy as np

import pandas as pd

import math



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt



from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px
healthsysdf = pd.read_csv('../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv') #load first dataset



healthsysdf = healthsysdf.drop(columns = 'Province_State') #drop useless columns

healthsysdf = healthsysdf.drop(columns = 'Country_Region')

healthsysdf['Total_Gov_Spend'] = healthsysdf.apply(lambda row: (row.Health_exp_pct_GDP_2016 / 100) * row.Health_exp_public_pct_2016, axis = 1) #calculate total government spending

healthsysdf['Outofpocket_Spend'] = healthsysdf.apply(lambda row: (row.Health_exp_pct_GDP_2016 / 100) * row.Health_exp_out_of_pocket_pct_2016, axis = 1) #calculate total out of pocket spending

healthsysdf['Other_Spend'] = healthsysdf.apply(lambda row: row.Health_exp_pct_GDP_2016 - row.Total_Gov_Spend - row.Outofpocket_Spend, axis = 1)





countrycodes = ['AFG', 'ALB', 'DZA', 'AND', 'AGO', 'ATG', 'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BTN', 'BOL', 'BIH', 'BWA', 'BRA',

                'BRN', 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', '', 'CAF', 'TCD', '', 'CHL', 'CHN', '', '', 'COL', 'COM', 'COD', 'COG', 'CRI', 'CIV', 'HRV', 'CUB', 'CYP',

                'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 'ETH', '', 'FJI', 'FIN', 'FRA', '', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GRC', '',

                'GRD', '', 'GTM', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', '', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR', 'KAZ', 'KEN', 'KIR', '', 'KOR', '', 'KWT',

                'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', '', '', 'LTU', 'LUX', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MAR',

                'MOZ', 'MMR', 'NAM', 'NPL', 'NLD', '', 'NZL', 'NGA', 'NER', 'NGA', 'MKD', '', 'NOR', 'OMN', 'PAK', 'PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', '', 'QAT', 'ROU', 'RUS',

                'RWA', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP', '', 'SVK', 'SVN', 'SLB', '', 'ZAF', '', 'ESP', 'LKA', 'KNA', 'LCA', '', 'VCT', 'SDN', 'SUR', 'SWE', 'CHE', '', 'TJK',

                'TZA', 'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', '', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT', 'VEN', 'VNM', '', '', 'YEM', 'ZMB', 'ZWE']



healthsysdf['Country_Codes'] = countrycodes #add country codes for use in map
bginfo = pd.read_csv('../input/undata-country-profiles/country_profile_variables.csv') #load second dataset

bginfo.rename(columns = {'country':'World_Bank_Name'}, inplace=True) #rename dataset to make combining easy



bginfo = bginfo.replace({'United States of America':'United States', 'Viet Nam': 'Vietnam'})



healthsysdf = healthsysdf.replace({'Yemen, Rep.': 'Yemen'})



healthsysdf = pd.merge(healthsysdf, bginfo, on='World_Bank_Name', how='outer') #combining datasets



healthsysdf = healthsysdf.dropna(thresh=3) #drop countries with little data



# Get the countries with GDP set below 0 and drop them from dataset

badgdp = healthsysdf[ healthsysdf['GDP: Gross domestic product (million current US$)'] < 0 ].index

healthsysdf.drop(badgdp , inplace=True)



# Create smaller regional groupings

healthsysdf.replace({'SouthernAsia':'Asia', 'WesternAsia':'Asia', 'EasternAsia':'Asia','CentralAsia':'Asia', 'South-easternAsia':'Asia',

                     'WesternEurope':'Europe', 'SouthernEurope':'Europe', 'EasternEurope':'Europe', 'NorthernEurope':'Europe',

                     'NorthernAfrica':'Africa', 'MiddleAfrica':'Africa', 'WesternAfrica':'Africa', 'EasternAfrica':'Africa', 'SouthernAfrica':'Africa',

                     'SouthAmerica':'Americas', 'Caribbean':'Americas', 'CentralAmerica':'Americas', 'NorthernAmerica':'Americas',

                     'Polynesia':'Oceania', 'Melanesia':'Oceania', 'Micronesia':'Oceania'}, inplace=True )
total_exp = healthsysdf.sort_values('Health_exp_pct_GDP_2016', ascending = False)

top_ten_exp = total_exp.head(10)

total_exp = total_exp.sort_values('Health_exp_pct_GDP_2016')

low_ten_exp = total_exp.head(10)



fig = make_subplots(rows=1, cols=2, shared_yaxes=True)



fig.add_trace(

    go.Bar(x=top_ten_exp['World_Bank_Name'], y=top_ten_exp['Health_exp_pct_GDP_2016']),

    row=1, col=1

)



fig.add_trace(

    go.Bar(x=low_ten_exp['World_Bank_Name'], y=low_ten_exp['Health_exp_pct_GDP_2016']),

    row=1, col=2

)





fig.update_layout(

    title={

        'text': "Ten highest and lowest spenders",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    plot_bgcolor= 'white',

    paper_bgcolor= 'white',

    yaxis_title="% of GDP spent on healthcare",

    showlegend=False,

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()
import plotly.graph_objects as go

import pandas as pd



fig = go.Figure(data=go.Choropleth(

    locations = healthsysdf['Country_Codes'],

    z = healthsysdf['Health_exp_pct_GDP_2016'],

    text = healthsysdf['World_Bank_Name'],

    colorscale = 'blues',

    autocolorscale=False,

    colorbar_tickprefix = '% ',

    marker_line_color='darkgray',

    marker_line_width=0.5,

))



fig.update_layout(

    title_text='Percentage of GDP spent on Healthcare',

    font=dict(

        family="Courier New, monospace",

        size=14),

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
fig = go.Figure(data=go.Choropleth(

    locations = healthsysdf['Country_Codes'],

    z = healthsysdf['Total_Gov_Spend'],

    text = healthsysdf['World_Bank_Name'],

    colorscale = 'blues',

    autocolorscale=False,

    colorbar_tickprefix = '% ',

    marker_line_color='darkgray',

    marker_line_width=0.5,

))



fig.update_layout(

    title_text='Government Spending on Healthcare',

    font=dict(

        family="Courier New, monospace",

        size=14),

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
fig = go.Figure(data=go.Choropleth(

    locations = healthsysdf['Country_Codes'],

    z = healthsysdf['per_capita_exp_PPP_2016'],

    text = healthsysdf['World_Bank_Name'],

    colorscale = 'blues',

    autocolorscale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

))



fig.update_layout(

    title_text='Healthcare Spending per Capita',

    font=dict(

        family="Courier New, monospace",

        size=14),

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
g8_list = ['Canada', 'United Kingdom', 'United States', 'Russian Federation', 'Germany', 'France', 'Japan', 'China']



g8_sub = healthsysdf.loc[healthsysdf['World_Bank_Name'].isin(g8_list)]

g8_sub = g8_sub.sort_values('Health_exp_pct_GDP_2016', ascending=False)



fig = go.Figure()

fig.add_trace(go.Bar(

    x=g8_sub['World_Bank_Name'],

    y=g8_sub['Health_exp_pct_GDP_2016'],

    name='Total Spending',

    marker_color='darkblue'

))

fig.add_trace(go.Bar(

    x=g8_sub['World_Bank_Name'],

    y=g8_sub['Total_Gov_Spend'],

    name='Government Spending',

    marker_color='mediumaquamarine'

))

fig.add_trace(go.Bar(

    x=g8_sub['World_Bank_Name'],

    y=g8_sub['Outofpocket_Spend'],

    name='Private (out of pocket) Spending',

    marker_color='lightsteelblue'

))

fig.add_trace(go.Bar(

    x=g8_sub['World_Bank_Name'],

    y=g8_sub['Other_Spend'],

    name='Other',

    marker_color='grey'

))



fig.update_layout(

    barmode='group',

    title={

        'text': "G8 Healthcare spending",

        'y':0.9,

        'x':0.4,

        'xanchor': 'center',

        'yanchor': 'top'},

    plot_bgcolor= 'white',

    paper_bgcolor= 'white',

    yaxis_title="% of GDP spent on healthcare",

    showlegend=True,

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()

interest_sub_list = ['Norway', 'Ireland', 'Netherlands', 'Switzerland', 'Brazil', 'Argentina', 'Mexico', 'Algeria', 'Namibia', 'Rwanda', 'South Africa', 'Indonesia', 'India', 'Australia']



interest_sub = healthsysdf.loc[healthsysdf['World_Bank_Name'].isin(interest_sub_list)]

interest_sub = interest_sub.sort_values('Health_exp_pct_GDP_2016', ascending=False)



fig = go.Figure()

fig.add_trace(go.Bar(

    x=interest_sub['World_Bank_Name'],

    y=interest_sub['Health_exp_pct_GDP_2016'],

    name='Total Spending',

    marker_color='darkblue'

))

fig.add_trace(go.Bar(

    x=interest_sub['World_Bank_Name'],

    y=interest_sub['Total_Gov_Spend'],

    name='Government Spending',

    marker_color='mediumaquamarine'

))

fig.add_trace(go.Bar(

    x=interest_sub['World_Bank_Name'],

    y=interest_sub['Outofpocket_Spend'],

    name='Private (out of pocket) Spending',

    marker_color='lightsteelblue'

))

fig.add_trace(go.Bar(

    x=interest_sub['World_Bank_Name'],

    y=interest_sub['Other_Spend'],

    name='Other',

    marker_color='grey'

))



fig.update_layout(

    barmode='group',

    title={

        'text': "Healthcare spending",

        'y':0.9,

        'x':0.4,

        'xanchor': 'center',

        'yanchor': 'top'},

    plot_bgcolor= 'white',

    paper_bgcolor= 'white',

    yaxis_title="% of GDP spent on healthcare",

    showlegend=True,

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()
healthsysdf = healthsysdf.dropna(subset=['GDP: Gross domestic product (million current US$)'])

size=healthsysdf['GDP: Gross domestic product (million current US$)']

sizeref = 2.*max(healthsysdf['GDP: Gross domestic product (million current US$)'])/(100**2)



fig = px.scatter(x=healthsysdf['per_capita_exp_PPP_2016'], y=healthsysdf['Physicians_per_1000_2009-18'],                 

                 size=size, color=healthsysdf['Region'],

                 hover_name=healthsysdf['World_Bank_Name'])



# Tune marker appearance and layout

fig.update_traces(mode='markers', marker=dict(sizemode='area',

                                              sizeref=sizeref, line_width=2))



fig.update_layout(

    title={

        'text': "Spending vs Physicians per 1000 people",

        'y':0.9,

        'x':0.4,

        'xanchor': 'center',

        'yanchor': 'top'},

    plot_bgcolor= 'white',

    paper_bgcolor= 'white',

    xaxis_title="Healthcare spending per capita",

    yaxis_title="Physicians per 1000 people",

    showlegend=True,

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig.show()