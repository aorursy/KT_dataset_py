# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

trade=pd.read_csv('/kaggle/input/global-commodity-trade-statistics/commodity_trade_statistics_data.csv', low_memory=False)

trade_df = trade[trade.country_or_area == "EU-28"]

trade_df.head()
tradeImport = trade_df[(trade_df.flow == 'Import') & (trade_df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

tradeExport = trade_df[(trade_df.flow == 'Export') & (trade_df.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')



trace1 = go.Bar(

                x = tradeImport.year,

                y = tradeImport.trade_usd,

                name = "Europe Import",

                marker = dict(color = 'rgba(102, 216, 137, 0.8)'),

)



trace2 = go.Bar(

               x = tradeExport.year,

                y = tradeExport.trade_usd,

                name = "Europe Export",

                marker = dict(color = 'rgba(224, 148, 215, 0.8)'),

)

data = [trace1, trace2]

layout = {

    'xaxis': {'title': 'Year 1990-2016'},

    'yaxis': {'title': 'Trade of Import & Export in Europe (USD)'},

    'barmode': 'group',

    'title': 'Import and Export in Europe'

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)
temp = trade_df[(trade_df.year==2000) & (trade_df.flow=='Import')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]

trade_2000import = temp.sort_values(by="trade_usd",  ascending=True)

trace1 = go.Bar(

                x = trade_2000import.trade_usd,

                y = trade_2000import.commodity,

                marker = dict(color = 'rgba(152, 213, 245, 0.8)'),

                orientation = 'h'

)





data = [trace1]

layout = {

#     'xaxis': {'title': 'Trade in USD'},

    'yaxis': {'automargin':True,},

    'title': "Top 10 Commodities in Europe Import Trade (USD), 2000"

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)



temp1 = trade_df[(trade_df.year==2015) & (trade_df.flow=='Import')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]

trade_2015import = temp1.sort_values(by="trade_usd",  ascending=True)

trace1 = go.Bar(

                x = trade_2015import.trade_usd,

                y = trade_2015import.commodity.tolist(),

                marker = dict(color = 'rgba(249, 205, 190, 0.8)'),

                orientation = 'h'

)





data = [trace1]

layout = {

#  'xaxis': {'title': 'Trade in USD'},

    'yaxis': {'automargin':True,},

    'title': "Top 10 Commodities in Europe Import Trade (USD), 2015"

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)
petro = trade_df[(trade_df.comm_code == '270900') & (trade_df.flow == 'Import')]

urea = trade_df[(trade_df.comm_code == '310210') & (trade_df.flow == 'Import')]

iron = trade_df[(trade_df.comm_code == '260111') & (trade_df.flow == 'Import')]



trace1 = go.Scatter(

    x = petro.year,

    y = petro.trade_usd,

    mode = "lines+markers",

    name = "Petroleum oils",

    marker = dict(color = 'rgba(263, 50, 30, 0.8)')

)



trace2 = go.Scatter(

    x = urea.year,

    y = urea.trade_usd,

    mode = "lines+markers",

    name = "Urea",

    marker = dict(color = 'rgba(130, 70, 0, 0.8)')

)



trace3 = go.Scatter(

    x = iron.year,

    y = iron.trade_usd,

    mode = "lines+markers",

    name = "Iron ore",

    marker = dict(color = 'rgba(42,53, 120, 0.8)')

)





data = [trace1, trace2, trace3]

layout = dict(title = "Some Commodities' value in Europe Import Trade (USD)",

              xaxis= dict(title= 'Year 2000-2015',ticklen= 5,zeroline= False),

              yaxis = {'title': 'Import trade value(USD)'}

 )

fig = dict(data = data, layout = layout)

iplot(fig)
temp = trade_df[(trade_df.year==2000) & (trade_df.flow=='Export')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]

trade_2000Export = temp.sort_values(by="trade_usd",  ascending=True)

trace1 = go.Bar(

                x = trade_2000Export.trade_usd,

                y = trade_2000Export.commodity,

                marker = dict(color = 'rgba(21, 31, 39, 0.8)'),

                orientation = 'h'

)





data = [trace1]

layout = {

#     'xaxis': {'title': 'Trade in USD'},

    'yaxis': {'automargin':True,},

    'title': "Top 10 Commodities in Europe Export Trade (USD), 2000"

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)



temp1 = trade_df[(trade_df.year==2015) & (trade_df.flow=='Export')].sort_values(by="trade_usd",  ascending=False).iloc[1:11, :]

trade_2015Export = temp1.sort_values(by="trade_usd",  ascending=True)

trace1 = go.Bar(

                x = trade_2015Export.trade_usd,

                y = trade_2015Export.commodity,

                marker = dict(color = 'rgba(125, 121, 80, 0.8)'),

                orientation = 'h'

)





data = [trace1]

layout = {

#     'xaxis': {'title': 'Trade in USD'},

    'yaxis': {'automargin':True,},

    'title': "Top 10 Commodities in Europe Export Trade (USD), 2015"

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)



toys = trade_df[(trade_df.comm_code == '950390') & (trade_df.flow == 'Export')]

maize = trade_df[(trade_df.comm_code == '100590') & (trade_df.flow == 'Export')]

light = trade_df[(trade_df.comm_code == '940540') & (trade_df.flow == 'Export')]



trace1 = go.Scatter(

    x = toys.year,

    y = toys.trade_usd,

    mode = "lines+markers",

    name = "toys",

    marker = dict(color = 'rgba(55, 96, 20, 0.8)')

)



trace2 = go.Scatter(

    x = maize.year,

    y = maize.trade_usd,

    mode = "lines+markers",

    name = "maize",

    marker = dict(color = 'rgba(241, 130, 133, 0.8)')

)



trace3 = go.Scatter(

    x = light.year,

    y = light.trade_usd,

    mode = "lines+markers",

    name = "light",

    marker = dict(color = 'rgba(130, 241, 140, 0.8)')

)





data = [trace1, trace2, trace3]

layout = dict(title = "Some Commodities' value in  Europe Export Trade (USD)",

              xaxis= dict(title= 'Year 2000-2015',ticklen= 5,zeroline= False),

              yaxis = {'title': 'Export trade value(USD)'}

 )

fig = dict(data = data, layout = layout)

iplot(fig)
Europe_trade = trade_df[trade_df.comm_code!= 'TOTAL'].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

wd_trade = trade[(trade.year >1991) & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

# cn_trade.shape



trace0 = {

    'x': Europe_trade.year,

    'y': Europe_trade.trade_usd,

    'name': "Europe",

    'type': 'bar',

    'marker': {'color':'rgba(129, 239, 208, 0.8)'}

}

trace1 = {

    'x': wd_trade.year,

    'y': wd_trade.trade_usd,

    'name': "World",

    'type': 'bar',

    'marker': {'color':'rgba(155, 174, 202, 0.8)'}

}



data = [trace0, trace1]

layout = {

    'xaxis': {'title': 'Year 1992-2016'},

    'yaxis': {'title': 'Value of Trade in USD'},

    'barmode': 'relative',

    'title': 'World vs Europe: Value of Trade'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)



# ratio

trace3 = go.Scatter(

                    x = Europe_trade.year,

                    y = Europe_trade.trade_usd/wd_trade.trade_usd*100,

                    mode = "lines+markers",

                    name = "Ratio of Europe-28/World",

                    marker = dict(color = 'rgba(45, 15, 104, 0.8)')

)

data2 = [trace3]

layout2 = dict(title = 'Percentage of Europe Trade in World Trade (%)',

              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),

              yaxis = {'title': 'Percentage (%)'}

 )

fig2 = dict(data = data2, layout = layout2)

iplot(fig2)
Trade_importExport = trade_df[trade_df.comm_code!= 'TOTAL'].groupby(['year'],as_index=False)['weight_kg'].agg('sum')

world_importExport = trade[(trade.year >1991) & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['weight_kg'].agg('sum')



trace1 = go.Bar(

                x = world_importExport.year,

                y = world_importExport.weight_kg,

                name = "World",

                marker = dict(color = 'rgba(104, 206, 245, 0.8)'),

)



trace2 = go.Bar(

                x = Trade_importExport.year,

                y = Trade_importExport.weight_kg,

                name = "Europe-28",

                marker = dict(color = 'rgba(25, 24, 12, 0.8)'),

)

data = [trace1, trace2]

layout = {

    'xaxis': {'title': 'Year 1992-2016'},

    'yaxis': {'title': 'Import & Export in Weight (kg)'},

    'barmode': 'group',

    'title': 'World vs Europe: Import & Export in Weight'

}

fig = go.Figure(data = data, layout = layout)

iplot(fig)



# ratio

trace3 = go.Scatter(

                    x = Trade_importExport.year,

                    y = Trade_importExport.weight_kg/world_importExport.weight_kg*100,

                    mode = "lines+markers",

                    name = "Ratio of Europe/World",

                    marker = dict(color = 'rgba(84, 222, 90, 0.8)')

)

data2 = [trace3]

layout2 = dict(title = 'Percentage of Europe\'s Import & Export in World (%)',

              xaxis= dict(title= 'Year 1992-2016',ticklen= 5,zeroline= False),

              yaxis = {'title': 'Percentage (%)'}

 )

fig2 = dict(data = data2, layout = layout2)

iplot(fig2)

USA_trade = trade[(trade.country_or_area == "USA") & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

JAPAN_trade = trade[(trade.country_or_area == "Japan") & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

CHINA_trade = trade[(trade.country_or_area == "China") & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

INDIA_trade = trade[(trade.country_or_area == "India") & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')

EUR_trade = trade[(trade.country_or_area == "EU-28") & (trade.comm_code!= 'TOTAL')].groupby(['year'],as_index=False)['trade_usd'].agg('sum')



EUR_2000 = int(EUR_trade[EUR_trade.year==2000].iloc[0][1])

USA_2000 = int(USA_trade[USA_trade.year==2000].iloc[0][1])

JAP_2000 = int(JAPAN_trade[JAPAN_trade.year==2000].iloc[0][1])

CHINA_2000 = int(CHINA_trade[CHINA_trade.year==2000].iloc[0][1])

INDIA_2000 = int(INDIA_trade[INDIA_trade.year==2000].iloc[0][1])

ot_2000 = int(wd_trade[wd_trade.year==2000].iloc[0][1]) - EUR_2000 - USA_2000 - JAP_2000 - CHINA_2000 - INDIA_2000



EUR_2015 = int(EUR_trade[EUR_trade.year==2015].iloc[0][1])

USA_2015 = int(USA_trade[USA_trade.year==2015].iloc[0][1])

JAP_2015 = int(JAPAN_trade[JAPAN_trade.year==2015].iloc[0][1])

CHINA_2015 = int(CHINA_trade[CHINA_trade.year==2015].iloc[0][1])

INDIA_2015 = int(INDIA_trade[INDIA_trade.year==2015].iloc[0][1])

ot_2015 = int(wd_trade[wd_trade.year==2015].iloc[0][1]) - EUR_2015 - USA_2015 - JAP_2015 - CHINA_2015 - INDIA_2015



labels = ['Europe','USA','Japan','China','India','Others']

colors = ['#f18285', '#86e48f', '#e8a2d8', '#fff76e','#47B39C','#FFC154']



#####

trace = go.Pie(labels=labels, values=[EUR_2000, USA_2000, JAP_2000, CHINA_2000, INDIA_2000, ot_2000],

               marker=dict(colors=colors,  line=dict(color='#000', width=2)) )

layout = go.Layout(

    title='2000 Import & Export Trade in USD',

)

fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='basic_pie_chart')



######

trace1 = go.Pie(labels=labels, values=[EUR_2015, USA_2015, JAP_2015, CHINA_2015, INDIA_2015, ot_2015],

               marker=dict(colors=colors,  line=dict(color='#000', width=2)) )

layout1 = go.Layout(

    title='2015 Import & Export Trade in USD',

)



fig1 = go.Figure(data=[trace1], layout=layout1)

iplot(fig1, filename='basic_pie_chart1')