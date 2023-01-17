# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#  source: http://www.un.org/en/development/desa/policy/wesp/wesp_current/2014wesp_country_classification.pdf
developed_countries = ["Austria", "Belgium", "Denmark", "France", "Germany", "Greece", "Ireland", 
              "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden", "United Kingdom",
              "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Estonia", "Hungary", "Latvia", "Lithuania",
              "Malta", "Poland", "Romania", "Slovakia", "Slovenia", "Iceland", "Norway", "Switzerland",
              "Australia", "Canada", "Japan", "New Zealand", "USA","Canada", "Japan", "France", "Germany"]
import os
print(os.listdir("../input/carbon-regression"))

# Any results you write to the current directory are saved as output.
countries = pd.read_csv("../input/mpi/MPI_national.csv")
print(countries.shape)
countries.head()

footprint = pd.read_csv("../input/ecological-footprint/countries.csv")
print(footprint.shape)
footprint.head()
trade = pd.read_csv("../input/global-commodity-trade-statistics/commodity_trade_statistics_data.csv")
trade = trade.rename(columns={'country_or_area': 'Country'})
print(trade.shape)
trade.head()
# recode levels
def recode(levels):
    if levels == 'Re-Export':
        return 'Export'
    else:
        return levels

trade['flow'] = trade['flow'].apply(recode)
country_trade = trade.groupby(["Country","flow"])["trade_usd"].sum().reset_index()
print(country_trade['flow'].value_counts())
# just query the exports leave out the imports
country_exptrade = country_trade.loc[(country_trade['flow'] == 'Export')].reset_index()
country_exptrade.head()
def recode_country(countries):
    if countries == 'Bolivia, Plurinational State of':
        return 'Bolivia'
    elif countries == 'Bolivia (Plurinational State of)':
        return 'Bolivia'
    elif countries == 'Bosnia and Herzegovina':
        return 'Bosnia Herzegovina'
    elif countries == 'Central African Rep.':
        return 'Central African Republic'
    elif countries == 'Congo, Democratic Republic of the':
        return 'Congo'
    elif countries == 'Congo, Republic of':
        return 'Congo'
    elif countries == 'Congo, Democratic Republic of':
        return 'Congo'
    elif countries == "Côte d'Ivoire":
        return 'CotedeIvoire'
    elif countries == "Cote d'Ivoire":
        return 'CotedeIvoire'
    elif countries == "Dominican Rep.":
        return "Dominican Republic"
    elif countries == "Fmr Fed. Rep. of Germany":
        return 'Germany'
    elif countries == "Korea, Democratic People's Republic of":
        return 'Korea, Republic of'
    elif countries == "Rep. of Korea":
        return 'Korea, Republic of'
    elif countries == "Lao People's Dem. Rep.":
        return "Lao People's Democratic Republic"
    elif countries == 'Macedonia, The former Yugoslav Republic of':
        return 'Macedonia'
    elif countries == "Macedonia TFYR":
        return 'Macedonia'
    elif countries == "TFYR of Macedonia":
        return 'Macedonia'
    elif countries == 'Moldova, Republic of':
        return 'Moldova'
    elif countries == 'Rep. of Moldova':
        return 'Moldova'
    elif countries == "Palestine, State ofa":
        return 'State of Palestine'
    elif countries == 'Solomon Islands':
        return 'Solomon Isds'
    elif countries == 'Iran, Islamic Republic of':
        return 'Iran'
    elif countries == 'Saint Vincent and the Grenadines':
        return 'Saint Vincent and Grenadines'
    elif countries == 'So. African Customs Union':
        return 'South Africa'
    elif countries == "Fmr Sudan":
        return 'Sudan'
    elif countries == 'Venezuela, Bolivarian Republic of':
        return 'Venezuela'
    elif countries == 'United Rep. of Tanzania':
        return 'Tanzania, United Republic of'
    elif countries == "United States":
        return 'USA'
    elif countries == "United States of America":
        return 'USA'
    else:
        return countries

countries['Country'] = countries['Country'].apply(recode_country)
footprint['Country'] = footprint['Country'].apply(recode_country)
country_exptrade['Country'] = country_exptrade['Country'].apply(recode_country)
MERGE = footprint.merge(countries, how="outer", on="Country")
print(MERGE.shape)
MERGE
MERGE2 = MERGE.merge(country_exptrade, how="outer", on="Country")
MERGE2 = MERGE2.drop(columns=['index', 'flow'])
MERGE2 = MERGE2.sort_values(['Country'], ascending=[True])
print(MERGE2.shape)
MERGE2.head()
MERGE2 = MERGE2.reset_index()
MERGE2['ExpTrade_Mill'] = MERGE2['trade_usd']/1000000
FIRST_WORLD = MERGE2[MERGE2['Country'].isin(developed_countries)].reset_index()
developing = set(MERGE2["Country"]).difference(set(developed_countries))
DEVELOPING = MERGE2[MERGE2['Country'].isin(developing)].reset_index()
MERGE2['ExpTrade_Mill'] = MERGE2['trade_usd']/1000000
MERGE2.columns
MERGE2['Carbon Footprint'].describe()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import math

df_2007 = MERGE2
df_2007['HDI'].fillna(0, inplace=True)
df_2007['ExpTrade_Mill'].fillna(0, inplace=True)
df_2007['Total Ecological Footprint'].fillna(0, inplace=True)
df_2007['GDP per Capita'].fillna(0, inplace=True)
df_2007["Population (millions)"].fillna(0, inplace=True)
df_2007["Carbon Footprint"].fillna(0, inplace=True)

#slope = 2.666051223553066e-05
hover_text = []
bubble_size = []

for index, row in df_2007.iterrows(): # each row of the df_2007 will be referenced
    hover_text.append(('Country: {country}<br>'+
                      'Total Ecological Footprint: {EcoFoot}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population (Millions): {pop}<br>'+
                      'Year: {year}').format(country=row['Country'],
                                            EcoFoot=row['Total Ecological Footprint'],
                                            gdp=row['GDP per Capita'],
                                            pop=row["Population (millions)"],
                                            year='2016'))
    #bubble_size.append(math.sqrt(row["Population (millions)"]*slope))
    bubble_size.append(row["Carbon Footprint"])

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2.*max(df_2007['size'])/(100**2)

trace0 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Africa'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Africa'],
    mode='markers',
    name='Africa',
    text=df_2007['text'][df_2007['Region'] == 'Africa'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Africa'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'North America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'North America'],
    mode='markers',
    name='North America',
    text=df_2007['text'][df_2007['Region'] == 'North America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'North America'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Middle East/Central Asia'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Middle East/Central Asia'],
    mode='markers',
    name='Middle East/Central Asia',
    text=df_2007['text'][df_2007['Region'] == 'Middle East/Central Asia'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Middle East/Central Asia'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Northern/Eastern Europe'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Northern/Eastern Europe'],
    mode='markers',
    name='Northern/Eastern Europe',
    text=df_2007['text'][df_2007['Region'] == 'Northern/Eastern Europe'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Northern/Eastern Europe'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Latin America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Latin America'],
    mode='markers',
    name='Latin America',
    text=df_2007['text'][df_2007['Region'] == 'Latin America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Latin America'],
        line=dict(
            width=2
        ),
    )
)

trace5 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Asia-Pacific'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Asia-Pacific'],
    mode='markers',
    name='Asia-Pacific',
    text=df_2007['text'][df_2007['Region'] == 'Asia-Pacific'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Asia-Pacific'],
        line=dict(
            width=2
        ),
    )
)

trace6 = go.Scatter(
     x=df_2007['HDI'][df_2007['Region'] == 'European Union'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'European Union'],
    mode='markers',
    name='European Union',
    text=df_2007['text'][df_2007['Region'] == 'European Union'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'European Union'],
        line=dict(
            width=2
        ),
    )
)
                      
trace7 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'nan'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'nan'],
    mode='markers',
    name='nan',
    text=df_2007['text'][df_2007['Region'] == 'nan'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'nan'],
        line=dict(
            width=2
        ),
    )
)
                      
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title='Impact on Carbon Footprint: Global Human Development v. Export Trade, 2008-2016',
    xaxis=dict(
        title='Human Development Index',
        gridcolor='rgb(255, 255, 255)',
        #range=[0, 1],
        #type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Cumulative Export Trade 2008-2016 (USD Millions)',
        gridcolor='rgb(255, 255, 255)',
        #range=[36.12621671352166, 91.72921793264332],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig) 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import math

df_2007 = DEVELOPING
df_2007['HDI'].fillna(0, inplace=True)
df_2007['ExpTrade_Mill'].fillna(0, inplace=True)
df_2007['Total Ecological Footprint'].fillna(0, inplace=True)
df_2007['GDP per Capita'].fillna(0, inplace=True)
df_2007["Population (millions)"].fillna(0, inplace=True)
df_2007["Carbon Footprint"].fillna(0, inplace=True)

#slope = 2.666051223553066e-05
hover_text = []
bubble_size = []

for index, row in df_2007.iterrows(): # each row of the df_2007 will be referenced
    hover_text.append(('Country: {country}<br>'+
                      'Total Ecological Footprint: {EcoFoot}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population (Millions): {pop}<br>'+
                      'Year: {year}').format(country=row['Country'],
                                            EcoFoot=row['Total Ecological Footprint'],
                                            gdp=row['GDP per Capita'],
                                            pop=row["Population (millions)"],
                                            year='2016'))
    #bubble_size.append(math.sqrt(row["Population (millions)"]*slope))
    bubble_size.append(row["Carbon Footprint"])

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2.*max(df_2007['size'])/(100**2)

trace0 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Africa'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Africa'],
    mode='markers',
    name='Africa',
    text=df_2007['text'][df_2007['Region'] == 'Africa'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Africa'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'North America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'North America'],
    mode='markers',
    name='North America',
    text=df_2007['text'][df_2007['Region'] == 'North America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'North America'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Middle East/Central Asia'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Middle East/Central Asia'],
    mode='markers',
    name='Middle East/Central Asia',
    text=df_2007['text'][df_2007['Region'] == 'Middle East/Central Asia'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Middle East/Central Asia'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Northern/Eastern Europe'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Northern/Eastern Europe'],
    mode='markers',
    name='Northern/Eastern Europe',
    text=df_2007['text'][df_2007['Region'] == 'Northern/Eastern Europe'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Northern/Eastern Europe'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Latin America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Latin America'],
    mode='markers',
    name='Latin America',
    text=df_2007['text'][df_2007['Region'] == 'Latin America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Latin America'],
        line=dict(
            width=2
        ),
    )
)

trace5 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Asia-Pacific'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Asia-Pacific'],
    mode='markers',
    name='Asia-Pacific',
    text=df_2007['text'][df_2007['Region'] == 'Asia-Pacific'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Asia-Pacific'],
        line=dict(
            width=2
        ),
    )
)

trace6 = go.Scatter(
     x=df_2007['HDI'][df_2007['Region'] == 'European Union'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'European Union'],
    mode='markers',
    name='European Union',
    text=df_2007['text'][df_2007['Region'] == 'European Union'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'European Union'],
        line=dict(
            width=2
        ),
    )
)
                      
trace7 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'nan'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'nan'],
    mode='markers',
    name='nan',
    text=df_2007['text'][df_2007['Region'] == 'nan'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'nan'],
        line=dict(
            width=2
        ),
    )
)
                      
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title='DEVELOPING NATIONS',
    xaxis=dict(
        title='Human Development Index',
        gridcolor='rgb(255, 255, 255)',
        #range=[0, 1],
        #type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Cumulative Export Trade 2008-2016 (USD Millions)',
        gridcolor='rgb(255, 255, 255)',
        range=[-4000000, 12000000],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig) 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import math

df_2007 = FIRST_WORLD
df_2007['HDI'].fillna(0, inplace=True)
df_2007['ExpTrade_Mill'].fillna(0, inplace=True)
df_2007['Total Ecological Footprint'].fillna(0, inplace=True)
df_2007['GDP per Capita'].fillna(0, inplace=True)
df_2007["Population (millions)"].fillna(0, inplace=True)
df_2007["Carbon Footprint"].fillna(0, inplace=True)

#slope = 2.666051223553066e-05
hover_text = []
bubble_size = []

for index, row in df_2007.iterrows(): # each row of the df_2007 will be referenced
    hover_text.append(('Country: {country}<br>'+
                      'Total Ecological Footprint: {EcoFoot}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population (Millions): {pop}<br>'+
                      'Year: {year}').format(country=row['Country'],
                                            EcoFoot=row['Total Ecological Footprint'],
                                            gdp=row['GDP per Capita'],
                                            pop=row["Population (millions)"],
                                            year='2016'))
    #bubble_size.append(math.sqrt(row["Population (millions)"]*slope))
    bubble_size.append(row["Carbon Footprint"])

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2.*max(df_2007['size'])/(100**2)

trace0 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Africa'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Africa'],
    mode='markers',
    name='Africa',
    text=df_2007['text'][df_2007['Region'] == 'Africa'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Africa'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'North America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'North America'],
    mode='markers',
    name='North America',
    text=df_2007['text'][df_2007['Region'] == 'North America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'North America'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Middle East/Central Asia'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Middle East/Central Asia'],
    mode='markers',
    name='Middle East/Central Asia',
    text=df_2007['text'][df_2007['Region'] == 'Middle East/Central Asia'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Middle East/Central Asia'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Northern/Eastern Europe'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Northern/Eastern Europe'],
    mode='markers',
    name='Northern/Eastern Europe',
    text=df_2007['text'][df_2007['Region'] == 'Northern/Eastern Europe'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Northern/Eastern Europe'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Latin America'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Latin America'],
    mode='markers',
    name='Latin America',
    text=df_2007['text'][df_2007['Region'] == 'Latin America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Latin America'],
        line=dict(
            width=2
        ),
    )
)

trace5 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Asia-Pacific'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Asia-Pacific'],
    mode='markers',
    name='Asia-Pacific',
    text=df_2007['text'][df_2007['Region'] == 'Asia-Pacific'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Asia-Pacific'],
        line=dict(
            width=2
        ),
    )
)

trace6 = go.Scatter(
     x=df_2007['HDI'][df_2007['Region'] == 'European Union'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'European Union'],
    mode='markers',
    name='European Union',
    text=df_2007['text'][df_2007['Region'] == 'European Union'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'European Union'],
        line=dict(
            width=2
        ),
    )
)
                      
trace7 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'nan'],
    y=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'nan'],
    mode='markers',
    name='nan',
    text=df_2007['text'][df_2007['Region'] == 'nan'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'nan'],
        line=dict(
            width=2
        ),
    )
)
                      
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title='FIRST-WORLD COUNTRIES: Impact on Carbon Footprint 2016',
    xaxis=dict(
        title='Human Development Index',
        gridcolor='rgb(255, 255, 255)',
        #range=[0, 1],
        #type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Cumulative Export Trade 2008-2016 (USD Millions)',
        gridcolor='rgb(255, 255, 255)',
        range=[-4000000, 19000000],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig) 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import math

df_2007 = MERGE2
df_2007['HDI'].fillna(0, inplace=True)
df_2007['ExpTrade_Mill'].fillna(0, inplace=True)
df_2007['Total Ecological Footprint'].fillna(0, inplace=True)
df_2007['GDP per Capita'].fillna(0, inplace=True)
df_2007["Population (millions)"].fillna(0, inplace=True)
df_2007["Carbon Footprint"].fillna(0, inplace=True)

#slope = 2.666051223553066e-05
hover_text = []
bubble_size = []

for index, row in df_2007.iterrows(): # each row of the df_2007 will be referenced
    hover_text.append(('Country: {country}<br>'+
                      'Total Ecological Footprint: {EcoFoot}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population (Millions): {pop}<br>'+
                      'Year: {year}').format(country=row['Country'],
                                            EcoFoot=row['Total Ecological Footprint'],
                                            gdp=row['GDP per Capita'],
                                            pop=row["Population (millions)"],
                                            year='2016'))
    #bubble_size.append(math.sqrt(row["Population (millions)"]*slope))
    bubble_size.append(row["HDI"])

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2.*max(df_2007['size'])/(10**2)

trace0 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Africa'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Africa'],
    mode='markers',
    name='Africa',
    text=df_2007['text'][df_2007['Region'] == 'Africa'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Africa'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'North America'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'North America'],
    mode='markers',
    name='North America',
    text=df_2007['text'][df_2007['Region'] == 'North America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'North America'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Middle East/Central Asia'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Middle East/Central Asia'],
    mode='markers',
    name='Middle East/Central Asia',
    text=df_2007['text'][df_2007['Region'] == 'Middle East/Central Asia'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Middle East/Central Asia'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Northern/Eastern Europe'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Northern/Eastern Europe'],
    mode='markers',
    name='Northern/Eastern Europe',
    text=df_2007['text'][df_2007['Region'] == 'Northern/Eastern Europe'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Northern/Eastern Europe'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Latin America'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Latin America'],
    mode='markers',
    name='Latin America',
    text=df_2007['text'][df_2007['Region'] == 'Latin America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Latin America'],
        line=dict(
            width=2
        ),
    )
)

trace5 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'Asia-Pacific'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Asia-Pacific'],
    mode='markers',
    name='Asia-Pacific',
    text=df_2007['text'][df_2007['Region'] == 'Asia-Pacific'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Asia-Pacific'],
        line=dict(
            width=2
        ),
    )
)

trace6 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'European Union'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'European Union'],
    mode='markers',
    name='European Union',
    text=df_2007['text'][df_2007['Region'] == 'European Union'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'European Union'],
        line=dict(
            width=2
        ),
    )
)
                      
trace7 = go.Scatter(
    x=df_2007['ExpTrade_Mill'][df_2007['Region'] == 'nan'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'nan'],
    mode='markers',
    name='nan',
    text=df_2007['text'][df_2007['Region'] == 'nan'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'nan'],
        line=dict(
            width=2
        ),
    )
)
                      
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title='Export Trade (2006-2016) Impact on Carbon Footprint',
    xaxis=dict(
        title='Cumulative Export 2008-2016 (USD Millions)',
        gridcolor='rgb(255, 255, 255)',
        #range=[0, 1],
        #type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Carbon Footprint ()',
        gridcolor='rgb(255, 255, 255)',
        range=[-5, 15],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig) 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import math

df_2007 = MERGE2
df_2007['HDI'].fillna(0, inplace=True)
df_2007['ExpTrade_Mill'].fillna(0, inplace=True)
df_2007['Total Ecological Footprint'].fillna(0, inplace=True)
df_2007['GDP per Capita'].fillna(0, inplace=True)
df_2007["Population (millions)"].fillna(0, inplace=True)
df_2007["Carbon Footprint"].fillna(0, inplace=True)

#slope = 2.666051223553066e-05
hover_text = []
bubble_size = []

for index, row in df_2007.iterrows(): # each row of the df_2007 will be referenced
    hover_text.append(('Country: {country}<br>'+
                      'Total Ecological Footprint: {EcoFoot}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population (Millions): {pop}<br>'+
                      'Year: {year}').format(country=row['Country'],
                                            EcoFoot=row['Total Ecological Footprint'],
                                            gdp=row['GDP per Capita'],
                                            pop=row["Population (millions)"],
                                            year='2016'))
    #bubble_size.append(math.sqrt(row["Population (millions)"]*slope))
    bubble_size.append(row["HDI"])

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2.*max(df_2007['size'])/(10**2)

trace0 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Africa'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Africa'],
    mode='markers',
    name='Africa',
    text=df_2007['text'][df_2007['Region'] == 'Africa'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Africa'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'North America'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'North America'],
    mode='markers',
    name='North America',
    text=df_2007['text'][df_2007['Region'] == 'North America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'North America'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Middle East/Central Asia'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Middle East/Central Asia'],
    mode='markers',
    name='Middle East/Central Asia',
    text=df_2007['text'][df_2007['Region'] == 'Middle East/Central Asia'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Middle East/Central Asia'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Northern/Eastern Europe'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Northern/Eastern Europe'],
    mode='markers',
    name='Northern/Eastern Europe',
    text=df_2007['text'][df_2007['Region'] == 'Northern/Eastern Europe'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Northern/Eastern Europe'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Latin America'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Latin America'],
    mode='markers',
    name='Latin America',
    text=df_2007['text'][df_2007['Region'] == 'Latin America'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Latin America'],
        line=dict(
            width=2
        ),
    )
)

trace5 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'Asia-Pacific'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'Asia-Pacific'],
    mode='markers',
    name='Asia-Pacific',
    text=df_2007['text'][df_2007['Region'] == 'Asia-Pacific'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'Asia-Pacific'],
        line=dict(
            width=2
        ),
    )
)

trace6 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'European Union'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'European Union'],
    mode='markers',
    name='European Union',
    text=df_2007['text'][df_2007['Region'] == 'European Union'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'European Union'],
        line=dict(
            width=2
        ),
    )
)
                      
trace7 = go.Scatter(
    x=df_2007['HDI'][df_2007['Region'] == 'nan'],
    y=df_2007["Carbon Footprint"][df_2007['Region'] == 'nan'],
    mode='markers',
    name='nan',
    text=df_2007['text'][df_2007['Region'] == 'nan'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['Region'] == 'nan'],
        line=dict(
            width=2
        ),
    )
)
                      
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    title='Human Development Impact on Carbon Footprint',
    xaxis=dict(
        title='Human Development Index',
        gridcolor='rgb(255, 255, 255)',
        range=[0.2, 1],
        #type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Carbon Footprint',
        gridcolor='rgb(255, 255, 255)',
        range=[-5, 15],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig) 
