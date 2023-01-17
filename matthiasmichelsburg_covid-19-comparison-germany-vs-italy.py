# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install chart_studio
import plotly.graph_objs as go

import chart_studio.plotly as py

import cufflinks



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from plotly.offline import iplot

cufflinks.go_offline()



# Set global theme

cufflinks.set_config_file(world_readable=True, theme='pearl')
def readData(fileName):

    dfOriginal = pd.read_csv(fileName)

    df = dfOriginal.iloc[:, 5:].T

    df.index = pd.to_datetime(df.index)

    df.columns = dfOriginal['Country/Region']

    df = df.groupby(df.columns, axis=1).sum()

    return df



dfConfirmed = readData('/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

dfDeaths = readData('/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
dfConfirmed
countries = ['Germany', 'Italy', 'France', 'United Kingdom', 'Spain', 'Japan', 'Korea, South', 'US']

country_color = {

    'Germany': '#000000', 

    'Italy': '#377d22', 

    'Spain': '#b62929'

} # creates dict of colors for highlights



def getTimeSeries(df):

    x = df.index

    y = df

    return (x, y)



def getTimeSeriesShifted(df, thresh):

    first_valid_loc = df.index.get_loc(df.where(df > thresh).first_valid_index())

    last_valid_loc = df.index.get_loc(df.where(df > thresh).last_valid_index())

    x=np.arange(last_valid_loc-first_valid_loc)

    y=df[first_valid_loc:last_valid_loc]

    return (x, y)



def plotTimeSeries(data, countries, country_color, getDataFunc):

    traces = []

    # now do the api cal'

    for country in countries:

        

        color = country_color.get(country, 'lightslategrey') # Get color of each line from country_color dict

        highlight = color != 'lightslategrey' # We only want to highlight a few traces, this will decide if a trace is highlighted or not



        x, y = getDataFunc(data[country])

        trace = go.Scatter(

                x=x,

                y=y,

                showlegend=highlight, # Show legend only if highlight



                name=country, 

                hoverinfo='name+x+y',

                mode='lines',



                line_color=color, # Line color

                opacity=0.8 if highlight else 0.4, # Different oppacity for highlighted lines



                line_shape='linear',

                line_smoothing=0.8,



                line_width=1.6 if highlight else 1 # Different width for highlighted lines     

            )

        traces.append(trace)

        

    fig = go.Figure(data=traces, layout=dict(

        yaxis_type='log',

        xaxis_title="Date",

        yaxis_title="Number of Cases")

    )

    return fig



figConfirmed = plotTimeSeries(dfConfirmed, countries, country_color, getTimeSeries)

figConfirmed.update_layout(title="Confirmed Cases")



figDeaths = plotTimeSeries(dfDeaths, countries, country_color, getTimeSeries)

figDeaths.update_layout(title="Fatalities")



figConfirmed = plotTimeSeries(dfConfirmed, countries, country_color, lambda x:getTimeSeriesShifted(x, 150))

figConfirmed.update_layout(title="Confirmed Cases", xaxis_title="Days after Outbreak")



figDeaths = plotTimeSeries(dfDeaths, countries, country_color, lambda x:getTimeSeriesShifted(x, 12))

figDeaths.update_layout(title="Fatalities", xaxis_title="Days after Outbreak")
dfOriginal = pd.read_excel('/kaggle/input/covid19-germany-badenwuerrtemberg/Tabelle_Coronavirus-Faelle-BW.xlsx', 

                    header=6, 

                    skipfooter=3)

dfBW = dfOriginal.iloc[:,1:].T

dfBW.columns = dfOriginal.iloc[:,0]

dfBW.index = pd.to_datetime(dfBW.index)

dfBW.sort_index(inplace=True)



dfBW.columns



landkreis_color = {

    'Reutlingen': '#000000', 

    'Tübingen': '#377d22', 

    'Heidelberg (Stadtkreis)': '#b62929',

    'BW': '#000000',

} # creates dict of colors for highlights



figBW = plotTimeSeries(dfBW, dfBW.columns, landkreis_color, getTimeSeries)

figBW.update_layout(title="Confirmed Cases - Baden-Württemberg", yaxis_type='linear')

figBW = plotTimeSeries(dfBW, dfBW.columns, landkreis_color, getTimeSeries)

figBW.update_layout(title="Confirmed Cases - Baden-Württemberg", yaxis_type='log')



dfBW.sum(axis=1)



figBW = plotTimeSeries(dfBW.sum(axis=1), ['BW'], {'BW': '#000000'}, getTimeSeries)

figBW.update_layout(title="Confirmed Cases - Baden-Württemberg", yaxis_type='log')