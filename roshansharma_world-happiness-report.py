!pip install bubbly
# for some basic operations

import numpy as np 

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')





# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

from bubbly.bubbly import bubbleplot



# for providing the path

import os

print(os.listdir("../input"))

data_2015 = pd.read_csv('../input/2015.csv')



data_2016 = pd.read_csv('../input/2016.csv')

data_2017 = pd.read_csv('../input/2017.csv')



data_2016.head()
# happiness score vs continents



plt.rcParams['figure.figsize'] = (15, 12)

sns.violinplot(data_2016['Happiness Score'], data_2016['Region'])

plt.show()

plt.rcParams['figure.figsize'] = (20, 15)

sns.heatmap(data_2017.corr(), cmap = 'copper', annot = True)



plt.show()



plt.rcParams['figure.figsize'] = (20, 15)



d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Western Europe']

sns.heatmap(d.corr(), cmap = 'Wistia', annot = True)



plt.show()



plt.rcParams['figure.figsize'] = (20, 15)



d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Eastern Asia']

sns.heatmap(d.corr(), cmap = 'Greys', annot = True)



plt.show()



plt.rcParams['figure.figsize'] = (20, 15)



d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'North America']

sns.heatmap(d.corr(), cmap = 'pink', annot = True)



plt.show()



plt.rcParams['figure.figsize'] = (20, 15)



d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Middle East and Northern Africa']



sns.heatmap(d.corr(), cmap = 'rainbow', annot = True)



plt.show()





plt.rcParams['figure.figsize'] = (20, 15)



d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Sub-Saharan Africa']

sns.heatmap(d.corr(), cmap = 'Blues', annot = True)



plt.show()



import warnings

warnings.filterwarnings('ignore')



figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Generosity', 

    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 

    x_title = "Happiness Score", y_title = "Generosity", title = 'Happiness vs Generosity vs Economy',

    x_logscale = False, scale_bubble = 1, height = 650)



py.iplot(figure, config={'scrollzoom': True})
import warnings

warnings.filterwarnings('ignore')



figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Trust (Government Corruption)', 

    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 

    x_title = "Happiness Score", y_title = "Trust", title = 'Happiness vs Trust vs Economy',

    x_logscale = False, scale_bubble = 1, height = 650)



py.iplot(figure, config={'scrollzoom': True})
import warnings

warnings.filterwarnings('ignore')



figure = bubbleplot(dataset = data_2016, x_column = 'Happiness Score', y_column = 'Health (Life Expectancy)', 

    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 

    x_title = "Happiness Score", y_title = "Health", title = 'Happiness vs Health vs Economy',

    x_logscale = False, scale_bubble = 1, height = 650)



py.iplot(figure, config={'scrollzoom': True})
import warnings

warnings.filterwarnings('ignore')



figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Family', 

    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 

    x_title = "Happiness Score", y_title = "Family", title = 'Happiness vs Family vs Economy',

    x_logscale = False, scale_bubble = 1, height = 650)



py.iplot(figure, config={'scrollzoom': True})
import plotly.figure_factory as ff



data = (

  {"label": "Happiness", "sublabel":"score",

   "range": [5, 6, 8], "performance": [5.5, 6.5], "point": [7]},

  {"label": "Economy", "sublabel": "score", "range": [0, 1, 2],

   "performance": [1, 1.5], "sublabel":"score","point": [1.5]},

  {"label": "Family","sublabel":"score", "range": [0, 1, 2],

   "performance": [1, 1.5],"sublabel":"score", "point": [1.3]},

  {"label": "Freedom","sublabel":"score", "range": [0, 0.3, 0.6],

   "performance": [0.3, 0.4],"sublabel":"score", "point": [0.5]},

  {"label": "Trust", "sublabel":"score","range": [0, 0.2, 0.5],

   "performance": [0.3, 0.4], "point": [0.4]}

)







fig = ff.create_bullet(

    data, titles='label', subtitles='sublabel', markers='point',

    measures='performance', ranges='range', orientation='v',

)

py.iplot(fig, filename='bullet chart from dict')
d2015 = data_2015['Region'].value_counts()



label_d2015 = d2015.index

size_d2015 = d2015.values





colors = ['aqua', 'gold', 'yellow', 'crimson', 'magenta']



trace = go.Pie(

         labels = label_d2015, values = size_d2015, marker = dict(colors = colors), name = '2015', hole = 0.3)



data = [trace]



layout1 = go.Layout(

           title = 'Regions')



fig = go.Figure(data = data, layout = layout1)

py.iplot(fig)



trace1 = [go.Choropleth(

               colorscale = 'Earth',

               locationmode = 'country names',

               locations = data_2017['Country'],

               text = data_2017['Country'], 

               z = data_2017['Generosity'],

               )]



layout = dict(title = 'Generosity',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)



data_2017[['Country', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
trace1 = [go.Choropleth(

               colorscale = 'Cividis',

               locationmode = 'country names',

               locations = data_2017['Country'],

               text = data_2017['Country'], 

               z = data_2017['Trust..Government.Corruption.'],

               )]



layout = dict(title = 'Trust in Governance',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)





data_2017[['Country', 'Trust..Government.Corruption.']].sort_values(by = 'Trust..Government.Corruption.',

                                                                     ascending = False).head(10)
trace1 = [go.Choropleth(

               colorscale = 'Portland',

               locationmode = 'country names',

               locations = data_2017['Country'],

               text = data_2017['Country'], 

               z = data_2017['Family'],

               )]



layout = dict(title = 'Family Satisfaction Index',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)





data_2017[['Country', 'Family']].sort_values(by = 'Family', ascending = False).head(10)

trace1 = [go.Choropleth(

               colorscale = 'Viridis',

               locationmode = 'country names',

               locations = data_2017['Country'],

               text = data_2017['Country'], 

               z = data_2017['Economy..GDP.per.Capita.'],

               )]



layout = dict(title = 'GDP in 2017',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)







data_2017[['Country', 'Economy..GDP.per.Capita.']].sort_values(by = 'Economy..GDP.per.Capita.',

            ascending = False).head(10)



trace1 = [go.Choropleth(

               colorscale = 'Picnic',

               locationmode = 'country names',

               locations = data_2017['Country'],

               text = data_2017['Country'], 

               z = data_2017['Freedom'],

               )]



layout = dict(title = 'Freedom Index',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)



data_2017[['Country', 'Freedom']].sort_values(by = 'Freedom', ascending = False).head(10)

trace1 = [go.Choropleth(

               colorscale = 'Electric',

               locationmode = 'country names',

               locations = data_2015['Country'],

               text = data_2015['Country'], 

               z = data_2015['Happiness Rank'],

               )]



layout = dict(title = 'Happiness Rank',

                  geo = dict(

                      showframe = True,

                      showocean = True,

                      showlakes = True,

                      showcoastlines = True,

                      projection = dict(

                          type = 'hammer'

        )))





projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 

               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 

               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 

               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]



buttons = [dict(args = ['geo.projection.type', y],

           label = y, method = 'relayout') for y in projections]



annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 

                    xref='paper', xanchor='right', showarrow=False )])





# Update Layout Object



layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])

layout[ 'annotations' ] = annot





fig = go.Figure(data = trace1, layout = layout)

py.iplot(fig)



data_2017[['Country','Happiness.Rank']].head(10)




































































































