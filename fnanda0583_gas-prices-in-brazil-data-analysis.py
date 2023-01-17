# Data exploration

import pandas as pd

import numpy as np



# To crate graphics

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.subplots import make_subplots





# To date

import datetime
# import data, creating a dataframe and first vizualization of dataframe

import os

os.chdir('../input')

df = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv', sep='\t',parse_dates=[1,2])

df.head()
# Shape of data frame

df.shape
# Removing index column

df_clean = df.drop("Unnamed: 0", axis=1)
# Rename the header

df_clean.rename(

    columns={

        "DATA INICIAL": "start_date",

        "DATA FINAL": "end_date",

        "REGIÃO": "region",

        "ESTADO": "state",

        "PRODUTO": "fuel",

        "NÚMERO DE POSTOS PESQUISADOS": "n_gas_stations",

        "UNIDADE DE MEDIDA": "unit",

        "PREÇO MÉDIO REVENDA": "avg_price",

        "DESVIO PADRÃO REVENDA": "sd_price",

        "PREÇO MÍNIMO REVENDA": "min_price",

        "PREÇO MÁXIMO REVENDA": "max_price",

        "MARGEM MÉDIA REVENDA": "avg_price_margin",

        "ANO": "year",

        "MÊS": "month",

        "COEF DE VARIAÇÃO DISTRIBUIÇÃO": "coef_dist",

        "PREÇO MÁXIMO DISTRIBUIÇÃO": "dist_max_price",

        "PREÇO MÍNIMO DISTRIBUIÇÃO": "dist_min_price",

        "DESVIO PADRÃO DISTRIBUIÇÃO": "dist_sd_price",

        "PREÇO MÉDIO DISTRIBUIÇÃO": "dist_avg_price",

        "COEF DE VARIAÇÃO REVENDA": "coef_price"

    },

    inplace=True

)



# Vizualization after the translation

df_clean.head()
# Information about data 

df_clean.info()
# Creating a for condition to convert the data type

for col in ['avg_price_margin', 'dist_avg_price', 'dist_sd_price', 'dist_min_price', 'dist_max_price', 'coef_dist']:

    

# to_numeric is the function that convert to float, and error=coerce is to ingnore NaN values

    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    

df_clean.info()
# Creating a dictionary to tranlate the regions names

regions = {"SUL":"South", "SUDESTE":"Southeast", "CENTRO OESTE":"Midwest", 

            "NORTE":"North", "NORDESTE":"Northeast"}



# Replace the names 

df_clean["region"] = df_clean.region.map(regions)



# Checking the translation

df_clean.region.value_counts()
# Creating a dictionary to tranlate the fuel names

fuels = {"ÓLEO DIESEL":"Diesel", "GASOLINA COMUM":"Regular Gasoline", "GLP":"LPG", 

            "ETANOL HIDRATADO":"Hydrous Ethanol", "GNV":"Natural Gas", "ÓLEO DIESEL S10":"Diesel S10"}



# Replace the names 

df_clean["fuel"] = df_clean.fuel.map(fuels)



# Checking the translation

df_clean.fuel.value_counts()
# Creating a new column. I will separate the date in month starting in day 1 

df_clean['month-year'] = pd.to_datetime(df_clean[['month','year']].assign(Day=1), format = "%m-%Y", errors='ignore')

df_clean.head()
# Customized float formatting

df_clean = df_clean.round(3)
# Checking the data type of new columns

df_clean.info()
# Creating a dataframe to number of observation by regions

df_NO = df_clean.region.value_counts().rename_axis('Region').reset_index(name='Number of observations')



# Creating a figure

fig = go.Figure()



# Creatin the bar graphic

fig.add_trace(go.Bar(x = df_NO['Region'], y = df_NO['Number of observations'], base=0,marker_color= 'navy',name='Number of observations'))



# Chart formatting

fig.update_layout(

    #title

    title=go.layout.Title(text="Number of observations by Regions",xref="paper",x=0),

    # Axis titles

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Region")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Observations")))





# Creating margin

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
# Creating a dataframe about number of observation by state

df_FR = df_clean.groupby("region")["fuel"].value_counts().unstack()



#Creating a list of fuels names

gas_list = df_FR.columns.tolist()



#Creating a list of regions names

region_list = df_FR.index.tolist()


fig = go.Figure()



for gas in gas_list:

    fig.add_trace(go.Bar(y=df_FR.index,x=df_FR[gas],name=gas,orientation='h'))

    

fig.update_layout(title=go.layout.Title(text="Number of observations by Regions",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Number of observations")),

    legend=dict(x=0, y=1.09))



# To fixe some bar separation

fig.update_traces(marker_line_color='black',marker_line_width=0.001)

# To stack bar

fig.update_layout(barmode='stack')

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
# Showing the units of fuels

df_clean.groupby("fuel")['unit'].value_counts()
# Creating a new dataframe to compare the prices of fuel in the same unit

df_fuels = df_clean[['start_date','end_date','region','state','fuel','unit','avg_price','month','year','month-year']]

df_fuels.head()
# Creating dictionary for conversion

# Units

event_dictionary_units ={'R$/l ' : 'R$/l', 'R$/13Kg' : 'R$/l', 'R$/m3' : 'R$/l'}



# Conversion factors

event_dictionary_conversion ={'R$/l' : 1 , 'R$/13Kg' : 0.00006153, 'R$/m3' : 0.001}



# Creating the new column

df_fuels['conversion'] = df_fuels['unit'].map(event_dictionary_conversion)



# Doing the convertion

df_fuels['avg_price'] = df_fuels.avg_price*df_fuels.conversion



# Changing the units

df_fuels['unit'] = df_fuels['unit'].map(event_dictionary_units)



# Checking the columns

df_fuels.tail()
# Creating a data frame of values in R$/l

df_fuels_avg = df_fuels.groupby(['year', 'fuel']).mean()['avg_price'].unstack().round(3)

df_fuels_avg

df_fuels_avg.columns.unique().tolist()
fig = go.Figure()



for gas in gas_list:

    fig.add_trace(go.Scatter(x = df_fuels_avg.index,y = df_fuels_avg[gas],mode = 'lines',name = gas))



fig.update_layout(title=go.layout.Title(text="Price of fuels by years",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title( text="Price (R$/l)")))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
# Creating dataframes for each fuel



df_GLP = df_clean.query('fuel in ["LPG"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)

df_GAS = df_clean.query('fuel in ["Regular Gasoline"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)

df_ETA = df_clean.query('fuel in ["Hydrous Ethanol"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)

df_GNV = df_clean.query('fuel in ["Natural Gas"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)

df_OLD = df_clean.query('fuel in ["Diesel"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)

df_OL10 = df_clean.query('fuel in ["Diesel S10"]').groupby(['month-year', 'region']).mean()['avg_price'].unstack().round(3)
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_GLP.index,y = df_GLP[region],mode = 'lines',name = region))



fig.update_layout(

    title=go.layout.Title(text="Price of LPG by Regions", xref="paper", x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years" )),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/13 Kg)"))

    ,xaxis_rangeslider_visible=True

    ,legend_orientation="h", legend=dict(x=0, y=1.03))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_GAS.index,y = df_GAS[region],mode = 'lines',name = region))



fig.update_layout(title=go.layout.Title(text="Price of Regular gasoline by Regions",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)"))

    ,xaxis_rangeslider_visible=True,

    legend_orientation="h", legend=dict(x=0, y=1.03))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_ETA.index,y = df_ETA[region],mode = 'lines',name = region))



fig.update_layout(title=go.layout.Title(text="Price of Hydrous Ethanol by Regions",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)"))

    ,xaxis_rangeslider_visible=True,

    legend_orientation="h", legend=dict(x=0, y=1.03))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_GNV.index,y = df_GNV[region],mode = 'lines',name = region))



fig.update_layout(title=go.layout.Title(text="Price of Natural gas by Regions", xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/m3)"))

    ,xaxis_rangeslider_visible=True,

    legend_orientation="h", legend=dict(x=0, y=1.03))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_OLD.index,y = df_OLD[region],mode = 'lines',name = region))



fig.update_layout(title=go.layout.Title(text="Price of Diesel by Regions",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)")),

    xaxis_rangeslider_visible=True,

    legend_orientation="h", legend=dict(x=0, y=1.03))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



fig.show()
fig = go.Figure()



for region in region_list:

    fig.add_trace(go.Scatter(x = df_OL10.index,y = df_OL10[region],mode = 'lines',name = region))



fig.update_layout(title=go.layout.Title(text="Price of Diesel S10 by Regions",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)")),

    xaxis_rangeslider_visible=True,

    legend_orientation="h", legend=dict(x=0, y=1.03))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





fig.show()
# Creating a data frame with values of regular gasoline in southeast in porcentage

df_GASP = df_clean.query('region in ["Southeast"] & fuel in ["Regular Gasoline"]').groupby(['month-year', 'region']).mean()['avg_price'].pct_change().unstack().round(3)

# Converting to porcentage

df_GASP = df_GASP.select_dtypes(exclude=['object']) * 100

# Creating a data frame of values of regular gasoline in southeast by year

df_GASP_S = df_clean.query('region in ["Southeast"] & fuel in ["Regular Gasoline"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)



# Creating a list of states 

sw_states = df_GASP_S.columns.tolist()



df_GASP_S
fig = go.Figure()



for state in sw_states:

    fig.add_trace(go.Scatter(x = df_GASP_S.index, y = df_GASP_S[state], mode = 'lines', name = state))



fig.update_layout(title=go.layout.Title(text="Price of Regular gasoline by Southeast States",xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)")),

    legend_orientation="h", legend=dict(x=0, y=1.1))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

# Creating the data frame with values of total variation in porcentage

df_GAS_DF = df_GASP_S.query('year in ["2004","2019"]')

df_GAS_DF = df_GAS_DF.pct_change().select_dtypes(exclude=['object']) * 100

df_GAS_DF = df_GAS_DF[1:2].round(3)

df_GAS_DF
fig = make_subplots(rows=2, cols=1, subplot_titles=("Over the years", "Total"),vertical_spacing = 0.08,row_heights=[0.7, 0.3])



for state in sw_states:

    fig.add_trace(go.Scatter(x = df_GASP_S.index, y = df_GASP_S[state], mode = 'lines', name = state), row=1, col=1)



fig.add_trace(go.Bar(x = df_GAS_DF.columns, y = df_GAS_DF.loc[2019], base=0,marker_color= 'sandybrown',name='Total Variation'),row=2, col=1)



fig.update_layout(height=900, width=1000, title_text="Price variation of Regular gasoline in the southeast states (2004-2019)")





fig.update_yaxes(title_text="Variation (%)", row=2, col=1)

fig.update_xaxes(title_text="States", row=2, col=1)

fig.update_yaxes(title_text="Variation (%)", row=1, col=1)



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)





#Creating a data frame with values of price variation in Rio de Janeiro in porcentage

df_GAS_RJ = df_clean.query('state in ["RIO DE JANEIRO"] & fuel in ["Regular Gasoline"]').groupby(['month-year', 'state']).mean()['avg_price'].unstack().round(3)

df_GASP_RJ = df_GAS_RJ.pct_change()

df_GASP_RJ = df_GASP_RJ.select_dtypes(exclude=['object']) * 100



# Creating a column do describe the variation

df_GASP_RJ['label'] = df_GASP_RJ['RIO DE JANEIRO'].apply(lambda x: 'Decrease' if x <= 0 else 'Increase')



# Spliting the values by leabels

df_GASP_RJ1 = df_GASP_RJ[df_GASP_RJ['label'] == 'Decrease'].round(3)

df_GASP_RJ2 = df_GASP_RJ[df_GASP_RJ['label'] == 'Increase'].round(3)
fig = make_subplots(rows=2, cols=1, subplot_titles=("Price", "Price variation (%)"),vertical_spacing = 0.1,row_heights=[0.6, 0.4])



# Creating a subgraphic tho show the variation 

fig.add_trace(go.Bar(x=df_GASP_RJ1.index, y=df_GASP_RJ1['RIO DE JANEIRO'],base=0,marker_color= 'red',name='Decrease'),row=2, col=1)

fig.add_trace(go.Bar(x=df_GASP_RJ2.index, y=df_GASP_RJ2['RIO DE JANEIRO'],base=0,marker_color= 'green',name='Increase'),row=2, col=1),



fig.add_trace(go.Scatter(x = df_GAS_RJ.index,y = df_GAS_RJ['RIO DE JANEIRO'],marker_color= 'navy', mode = 'lines',name = 'RIO DE JANEIRO'), row=1, col=1)



fig.update_layout(height=800, width=1000, title_text="Price variation of Regular gasoline in Rio de Janeiro (2004-2019)", legend_orientation="h", legend=dict(x=0, y=1.05))



fig.update_yaxes(title_text="Variation (%)", row=2, col=1)

fig.update_xaxes(title_text="Years", row=2, col=1)

fig.update_yaxes(title_text="Price(R$/l)", row=1, col=1)



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(zeroline=True, zerolinewidth=0.5, zerolinecolor='black')





fig.show()
# Creating a dataframe with data of natural gas in midwest region

df_GNVP_S = df_clean.query('region in ["Midwest"] & fuel in ["Natural Gas"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)



# Creating a list of states 

mw_states = df_GNVP_S.columns.tolist()



df_GNVP_S
# Creating a graphic 

fig = go.Figure()



for state in mw_states:

    fig.add_trace(go.Scatter(x = df_GNVP_S.index, y = df_GNVP_S[state], mode = 'lines', name = state))



fig.update_layout(title=go.layout.Title(text="Price of Natural gas by Midwest States",xref="paper",x=0,),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Years")),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/m3)")))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



#Creating a data frame with values of price variation in Mato Grosso do Sul in porcentage

df_GNV_MS = df_clean.query('state in ["MATO GROSSO DO SUL"] & fuel in ["Natural Gas"]').groupby(['month-year', 'state']).mean()['avg_price'].unstack().round(3)

df_GNVP_MS = df_GNV_MS.pct_change()

df_GNVP_MS = df_GNVP_MS.select_dtypes(exclude=['object']) * 100



# Creating a column do describe the variation

df_GNVP_MS['label'] = df_GNVP_MS['MATO GROSSO DO SUL'].apply(lambda x: 'Decrease' if x <= 0 else 'Increase')



# Spliting the values by leabels

df_GNVP_MS1 = df_GNVP_MS[df_GNVP_MS['label'] == 'Decrease'].round(3)

df_GNVP_MS2 = df_GNVP_MS[df_GNVP_MS['label'] == 'Increase'].round(3)
fig = make_subplots(rows=2, cols=1, subplot_titles=("Price", "Price variation (%)"),vertical_spacing = 0.1,row_heights=[0.6, 0.4])





fig.add_trace(go.Bar(x=df_GNVP_MS1.index, y=df_GNVP_MS1['MATO GROSSO DO SUL'],base=0,marker_color= 'red',name='Decrease'),row=2, col=1)

fig.add_trace(go.Bar(x=df_GNVP_MS2.index, y=df_GNVP_MS2['MATO GROSSO DO SUL'],base=0,marker_color= 'green',name='Increase'),row=2, col=1)



fig.add_trace(go.Scatter(x = df_GNV_MS.index,y = df_GNV_MS['MATO GROSSO DO SUL'],marker_color= 'navy',mode = 'lines',name = 'MATO GROSSO DO SUL'),row=1, col=1)



fig.update_layout(height=800, width=1000, title_text="Price variation of Natural Gas in Mato Grosso do Sul (2004-2019)", legend_orientation="h", legend=dict(x=-0.02, y=1.06))



fig.update_yaxes(title_text="Variation (%)", row=2, col=1)

fig.update_xaxes(title_text="Years", row=2, col=1)

fig.update_yaxes(title_text="Price(R$/m3)", row=1, col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



fig.update_yaxes(zeroline=True, zerolinewidth=0.5, zerolinecolor='black')





fig.show()
# Creating df for each fuels in 2019

df_GLP_D = df_clean.query('year==2019 & fuel in ["LPG"]')

df_GAS_D = df_clean.query('year==2019 & fuel in ["Regular Gasoline"]')

df_ETA_D = df_clean.query('year==2019 & fuel in ["Hydrous Ethanol"]')

df_GNV_D = df_clean.query('year==2019 & fuel in ["Natural Gas"]')

df_OLD_D = df_clean.query('year==2019 & fuel in ["Diesel"]')

df_OL10_D = df_clean.query('year==2019 & fuel in ["Diesel S10"]')
fig = go.Figure(data=[go.Box(y=df_GLP_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'LPG' )])



fig.update_layout(title=go.layout.Title(text="Box plot of LPG avarege price in 2019",xref="paper",x=0,),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/13Kg)")))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



fig.show()
fig = go.Figure(data=[go.Box(y=df_GNV_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'Natural Gas' )])



fig.update_layout(title=go.layout.Title(text="Box plot of Natural Gas avarege price in 2019",xref="paper",x=0,),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/m3)")))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



fig.show()
fig = go.Figure()



fig.add_trace(go.Box(y=df_GAS_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'Regular Gasoline' ))

fig.add_trace(go.Box(y=df_ETA_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'Hydrous Ethanol' ))

fig.add_trace(go.Box(y=df_OLD_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'Diesel' ))

fig.add_trace(go.Box(y=df_OL10_D['avg_price'],boxpoints='all', jitter=0.5,pointpos=-1.8, name = 'Diesel S10' ))



fig.update_layout(title=go.layout.Title(text="Box plot of fuels avarege price in 2019",xref="paper",x=0,),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Price (R$/l)")))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)



fig.show()
# Creating data frame of avarage prices in each year.

df_GLP_S = df_clean.query('year==2019 & fuel in ["LPG"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_GAS_S = df_clean.query('year==2019 & fuel in ["Regular Gasoline"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_ETA_S = df_clean.query('year==2019 & fuel in ["Hydrous Ethanol"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_GNV_S = df_clean.query('year==2019 & fuel in ["Natural Gas"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_OLD_S = df_clean.query('year==2019 & fuel in ["Diesel"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_OL10_S = df_clean.query('year==2019 & fuel in ["Diesel S10"]').groupby(['year', 'state']).mean()['avg_price'].unstack().round(3)

df_GLP_S = df_GLP_S.sort_values(by = df_GLP_S.index[0], axis=1)



fig = go.Figure()



fig.add_trace(go.Bar(y=df_GLP_S.columns,x=df_GLP_S.loc[2019],marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1),),name='GNV',orientation='h',))



fig.update_layout(height=700, width=800, title_text="Price of LPG in 2019 by states")

fig.update_xaxes(title_text='Price (R$/13 Kg)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
df_GAS_S = df_GAS_S.sort_values(by = df_GAS_S.index[0], axis=1)



fig = go.Figure()





fig.add_trace(go.Bar(y=df_GAS_S.columns, x=df_GAS_S.loc[2019],base=0,marker_color= 'tan',name='Gasolina Comum',orientation='h'))





fig.update_layout(height=700, width=800, title_text="Price of Regular Gasoline in 2019 by states")

fig.update_xaxes(title_text='Price (R$/l)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

df_ETA_S = df_ETA_S.sort_values(by = df_ETA_S.index[0], axis=1)



fig = go.Figure()



fig.add_trace(go.Bar(y=df_ETA_S.columns, x=df_ETA_S.loc[2019],base=0,marker_color= 'lightblue',name='Etanol Comum',orientation='h'))





fig.update_layout(height=700, width=800, title_text="Price of Hydrous ethanol in 2019 by states")

fig.update_xaxes(title_text='Price (R$/l)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
df_GNV_S = df_GNV_S.sort_values(by = df_GNV_S.index[0], axis=1)



fig = go.Figure()



fig.add_trace(go.Bar(y=df_GNV_S.columns, x=df_GNV_S.loc[2019],base=0,marker_color= 'lightcoral',name='GNV',orientation='h'))



fig.update_layout(height=700, width=800, title_text="Price of Natural gas in 2019 by states")

fig.update_xaxes(title_text='Price (R$/ 13Kg)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
df_OLD_S = df_OLD_S.sort_values(by = df_OLD_S.index[0], axis=1)



fig = go.Figure()



fig.add_trace(go.Bar(y=df_OLD_S.columns, x=df_OLD_S.loc[2019],base=0,marker_color= 'mediumvioletred',name='Óleo Diesel',orientation='h'))





fig.update_layout(height=700, width=800, title_text="Price of Diesel in 2019 by states")

fig.update_xaxes(title_text='Price (R$/l)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
df_OL10_S = df_OL10_S.sort_values(by = df_OL10_S.index[0], axis=1)



fig = go.Figure()





fig.add_trace(go.Bar(y=df_OL10_S.columns, x=df_OL10_S.loc[2019], base=0, marker_color= 'olive', name='Óleo Disel S10',orientation='h'))



fig.update_layout(height=700, width=800, title_text="Price of Diesel S10 in 2019 by states")

fig.update_xaxes(title_text='Price (R$/l)'),

fig.update_yaxes(tickfont=dict(size=10))



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
