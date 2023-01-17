# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

# word cloud library

from wordcloud import WordCloud



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/hfi_cc_2018.csv')
data.head(5)
# Filtering data to observe Middle Eastern and North Africaan Countries in the most recent year (2016) :

filter01 = data.year == 2016

filter02 = data.region == 'Middle East & North Africa'

data01 = data[filter01 & filter02]

data01.sample(5)
# First trace will be security of women in 2016: 

trace01 = go.Scatter(

                     x = data01['ISO_code'],

                     y = data01['pf_ss_women'],

                     mode = 'lines+markers',

                     name = 'Security of Women',

                     marker = dict(color = 'rgba(150, 20, 20, 0.5)'),

                     text= data01['countries'])

# Second trace will show us freedom of women movement in 2016:

trace02 = go.Scatter(

                     x = data01['ISO_code'],

                     y = data01['pf_movement_women'],

                     mode = 'lines+markers',

                     name = 'Freedom of Movement',

                     marker = dict(color = 'rgba(20, 150, 20, 0.5)'),

                     text= data01['countries'])

datanew = [trace01, trace02]

layoutnew = dict(title = 'Security of Women and Freedom of Women Movement Comparison in Middle East and Northern Africa',

                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))

fig = dict(data=datanew, layout=layoutnew)

py.iplot(fig)
# Filtering dataset to three different years (2008, 2013,2016)

data2008 = data[data.year == 2008]

data2008 = data2008[data2008.region == 'Latin America & the Caribbean']



data2013 = data[data.year == 2013]

data2013 = data2013[data2013.region == 'Latin America & the Caribbean']



data2016 = data[data.year == 2016]

data2016 = data2016[data2016.region == 'Latin America & the Caribbean']



# First trace is security of women in 2008:

trace2008 = go.Scatter(

                        x = data2008['ISO_code'],

                        y = data2008['pf_ss_women'],

                        mode = 'markers',

                        name = 'Women Security in 2008',

                        marker = dict(color='rgba(0,0,255,0.5)'),

                        text = data2008['countries'])



# Second trace is security of women in 2013:

trace2013 = go.Scatter(

                        x = data2013['ISO_code'],

                        y = data2013['pf_ss_women'],

                        mode = 'markers',

                        name = 'Women Security in 2013',

                        marker = dict(color='rgba(0,255,0,0.5)'),

                        text = data2013['countries'])



# Third trace is security of women in 2016:

trace2016 = go.Scatter(

                        x = data2016['ISO_code'],

                        y = data2016['pf_ss_women'],

                        mode = 'markers',

                        name = 'Women Security in 2016',

                        marker = dict(color='rgba(255,0,0,0.5)'),

                        text = data2016['countries'])

datanew = [trace2008, trace2013, trace2016]

layoutnew = dict(title = 'Security of Women in Latin America & the Caribbean Between 2008, 2013, 2016',

                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))

fig = dict(data=datanew, layout=layoutnew)

py.iplot(fig)
# Missing women in Middle East and North Africa:

trace01 = go.Bar(

                 x = data01['ISO_code'],

                 y = data01['pf_ss_women_missing'],

                 name = 'Missing Women',

                 marker = dict(color='rgba(130, 35, 45, 0.5)',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = data01.countries)

# Security Level of the Country:

trace02 = go.Bar(

                 x = data01['ISO_code'],

                 y = data01['pf_ss'],

                 name = 'Security of the Country',

                 marker = dict(color='rgba(45, 35, 145, 0.5)',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = data01.countries)

datanew = [trace01, trace02]

layoutnew = go.Layout(barmode='group', title='Missing Women Comparison')

fig = go.Figure(data=datanew, layout=layoutnew)

py.iplot(fig)
# Analysing security of women among Middle East and North African Countries:

fig = {

        'data': [ 

             {

                'values' : data01['pf_ss_women'],

                'labels' : data01['countries'],

                'domain' : {'x': [0, 1]},

                'name' : 'Security for Women',

                'hoverinfo' : 'label+percent+name',

                'hole' : 0.3,

                'type' : 'pie'

              },

             ],

         'layout' : {

                     'title' : 'Security of Women Among Middle East and N.Africa',

                     'annotations' : [

                                        { 'font' : {'size' : 20},

                                          'showarrow' : False,

                                          'text' : ' ',

                                          'x' : 0.20,

                                          'y' : 1

                                         },

                                      ]    

                     }

        }

py.iplot(fig)
# Filtering data to see only European Countries informations in 2016:

dataE = data[data.year == 2016]

dataEU = dataE[(dataE.region == 'Western Europe') + (dataE.region == 'Eastern Europe')]

dataEU = dataEU.dropna(subset=['pf_ss_women_missing'])

dataEU.sample(5)
colorEU = [float(each) for each in dataEU['pf_ss_women']]

sizeEU = [float(each) for each in dataEU['pf_ss_women']]



data05 = [

    {

        'x' : dataEU['pf_rol'],

        'y' : dataEU['pf_religion'],

        'mode' : 'markers',

        'marker' : { 'color': colorEU, 'size' : sizeEU, 'showscale' : True},

        'text' : dataEU.countries

    } 

]

layout05 = dict(title = 'Relationship Between Security of Women & Rule of Law & Freedom of Religion', 

                xaxis = dict(title = 'Rule of Law Index', ticklen = 4, zeroline = False),

                yaxis = dict(title = 'Freedom of Religion Index', ticklen = 4, zeroline = False))

fig = dict(data=data05, layout=layout05)

py.iplot(fig)
colorEU = [float(each) for each in dataEU['pf_ss_women']]



trace3D = go.Scatter3d(

    x = dataEU['pf_rol'],

    y = dataEU['pf_religion'],

    z = dataEU['pf_ss_women'],

    mode = 'markers',

    marker = dict(

        size = 7,

        color=colorEU

    )

)



data06 = [trace3D]

layout06 = go.Layout(

    title = 'Relationship Between Security of Women & Rule of Law & Freedom of Religion',

)

fig = go.Figure(data=data06, layout=layout06)

py.iplot(fig)
# Lets visualize the subject above with a histogram:

data2008 = data[(data.year == 2008) + (data.region == 'South Asia')]

data2008 = data2008.dropna(subset=['pf_ss_women'])

data2016 = data[(data.year == 2016) + (data.region == 'South Asia')]

data2016 = data2016.dropna(subset=['pf_ss_women'])



trace01 = go.Histogram(

                        x = data2008['pf_ss_women'],

                        opacity = 0.65,

                        name = 'Security of Women in 2008',

                        marker = dict(color='rgba(150,237,32,0.5)'),

                        text= data2008['countries'])

trace02 = go.Histogram(

                        x = data2016['pf_ss_women'],

                        opacity = 0.65,

                        name = 'Security of Women in 2016',

                        marker = dict(color='rgba(80,22,236,0.5)'))

dataHist = [trace01, trace02]

layoutHist = go.Layout(

                        title = 'Comparison of Security of Women in the Years 2008 and 2016',

                        xaxis = dict(title='Countries'))

fig = go.Figure(data = dataHist, layout = layoutHist)

py.iplot(fig)
region = data.region

plt.subplots(figsize = (8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(region))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
dataEF = data.filter(['year', 'ISO_code', 'countries', 'region', 'ef_score', 'ef_rank', 'hf_score', 'hf_rank'])

EastEU = dataEF[dataEF.region == 'Eastern Europe']

WestEU = dataEF[dataEF.region == 'Western Europe']
trace1 = go.Box(

                y = WestEU.ef_rank,

                name = 'Economic Freedom Rank of Western European Countries',

                marker = dict(color= 'rgb(78, 54, 23)')

                )

trace2 = go.Box(

                y = EastEU.ef_rank,

                name = 'Economic Freedom Rank of Eastern European Countries',

                marker = dict(color = 'rgb(20, 24, 79)')

                )

efdata = [trace1, trace2]

py.iplot(efdata)
dataocean = data[data.region == 'Oceania']

dataocean = dataocean.filter(['pf_ss_women', 'pf_ss', 'ef_rank'])

dataocean['index'] = np.arange(1, len(dataocean)+1)
import plotly.figure_factory as ff

fig = ff.create_scatterplotmatrix(dataocean, diag='box', index = 'index', colormap='Portland', colormap_type='cat', height=700, width=700)

py.iplot(fig)
dataset = data[data.year == 2016]

dataset = dataset.loc[:, ['year', 'countries', 'pf_ss_women']]

dataset.tail()
ssw = [dict(

    type = 'choropleth',

    locations = dataset['countries'],

    locationmode = 'country names',

    z = dataset['pf_ss_women'],

    text = dataset['countries'],

    colorscale = [[0,"rgb(5, 10, 172)"],[2,"rgb(40, 60, 190)"],[4,"rgb(70, 100, 245)"],\

            [6,"rgb(90, 120, 245)"],[8,"rgb(106, 137, 247)"],[10,"rgb(220, 220, 220)"]],

    autocolorscale = False,

    reversescale = True,

    marker = dict(line = dict(color = 'rgb(150,150,150)',width = 0.5 )),

    colorbar = dict(autotick=False, tickprefix= '', title='Security of Women'),

)]



layout = dict(

    title = 'Securtiy of Women in 2016',

    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))

)



fig = dict(data=ssw, layout=layout)

py.iplot( fig, validate=False, filename='security-of-women')