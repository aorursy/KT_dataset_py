import pandas as pd



import numpy as np



import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff





import cufflinks as cf

cf.set_config_file(offline=True)
df = pd.read_csv('../input/world_happiness_2018.csv')

df = df.loc[df['Year']==2008]

df = df.drop('Year',axis=1).reset_index()
url = 'http://worldhappiness.report/ed/2019/'



plotmap = [ dict(

        type = 'choropleth',

        locations = df['Country name'],

        locationmode = 'country names',

        z = df['Life Ladder'],

        text = df['Country name'],

        colorscale = 'Viridis',

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar =go.choropleth.ColorBar(

            title = 'Score',

            len=0.5

        ),

      ) ]



layout = dict(

    margin={"t": 0, "b": 0, "l": 0, "r": 0},

    geo = dict(

        showframe = True,

        showcoastlines = True,

        showocean = False,

        showcountries = True,

        oceancolor = '#26466D',

        projection = dict(

            type = 'mercator'

        )

    ),

     height=700,

     width=700,

    annotations = [go.layout.Annotation(

            x = 0.60,

            y = 0.09,

            xref = 'paper',

            yref = 'paper',

            text = 'Source: <a href="%s">World Happiness Report 2019</a>' % url,

            showarrow = False

        )]

)



fig = dict( data=plotmap, layout=layout )

iplot(fig)
df.drop(['index','Country name', 'Standard deviation of ladder by country-year',

       'Standard deviation/Mean of ladder by country-year', 'GINI index (World Bank estimate), average 2000-16',

       'gini of household income reported in Gallup, by wp5-year',

       'Most people can be trusted, Gallup',

       'Most people can be trusted, WVS round 1981-1984',

       'Most people can be trusted, WVS round 1989-1993',

       'Most people can be trusted, WVS round 1994-1998',

       'Most people can be trusted, WVS round 1999-2004',

       'Most people can be trusted, WVS round 2005-2009',

       'Most people can be trusted, WVS round 2010-2014'],axis=1).corr().iplot(

    kind='heatmap',colorscale='Greens',title="Feature Correlation Matrix", theme='white',

    tickangle=345, margin = {'l': 240})
Asia = ["Israel", "United Arab Emirates", "Singapore", "Thailand", "Taiwan Province of China",

                                   "Qatar", "Saudi Arabia", "Kuwait", "Bahrain", "Malaysia", "Uzbekistan", "Japan",

                                   "South Korea", "Turkmenistan", "Kazakhstan", "Turkey", "Hong Kong S.A.R., China", "Philippines",

                                   "Jordan", "China", "Pakistan", "Indonesia", "Azerbaijan", "Lebanon", "Vietnam",

                                   "Tajikistan", "Bhutan", "Kyrgyzstan", "Nepal", "Mongolia", "Palestinian Territories",

                                   "Iran", "Bangladesh", "Myanmar", "Iraq", "Sri Lanka", "Armenia", "India", "Georgia",

                                   "Cambodia", "Afghanistan", "Yemen", "Syria"]

Europe = ["Norway", "Denmark", "Iceland", "Switzerland", "Finland",

                                   "Netherlands", "Sweden", "Austria", "Ireland", "Germany",

                                   "Belgium", "Luxembourg", "United Kingdom", "Czech Republic",

                                   "Malta", "France", "Spain", "Slovakia", "Poland", "Italy",

                                   "Russia", "Lithuania", "Latvia", "Moldova", "Romania",

                                   "Slovenia", "North Cyprus", "Cyprus", "Estonia", "Belarus",

                                   "Serbia", "Hungary", "Croatia", "Kosovo", "Montenegro",

                                   "Greece", "Portugal", "Bosnia and Herzegovina", "Macedonia",

                                   "Bulgaria", "Albania", "Ukraine"]

Oceania = ["New Zealand", "Australia"]

North_America = ["Canada", "Costa Rica", "United States", "Mexico",  

                                   "Panama","Trinidad and Tobago", "El Salvador", "Belize", "Guatemala",

                                   "Jamaica", "Nicaragua", "Dominican Republic", "Honduras",

                                   "Haiti"]

South_America = ["Chile", "Brazil", "Argentina", "Uruguay",

                                   "Colombia", "Ecuador", "Bolivia", "Peru",

                                   "Paraguay", "Venezuela"]





def GetConti(country):

    if country in Asia:

        return "Asia"

    elif country in Europe:

        return "Europe"

    elif country in North_America:

        return "North America"

    elif country in South_America:

        return "South America"

    elif country in Oceania:

        return "Oceania"

    else:

        return "Africa"



df['Continent'] = df['Country name'].apply(lambda x: GetConti(x))
def scatter_plot(x, x_title):

    df.iplot(

        kind='scatter',

        mode='markers',

        x=x,

        y='Life Ladder',

        text='Country name',

        categories='Continent',

        xTitle=x_title,

        yTitle='Happiness Score',

        theme='white')

scatter_plot('Healthy life expectancy at birth', 'Healthy life expectancy')

scatter_plot('Log GDP per capita', 'GDP per capita')

scatter_plot('Perceptions of corruption', 'Perceptions of corruption')

scatter_plot('Social support', 'Social support')

scatter_plot('GINI index (World Bank estimate)', 'GINI index - Inequality')
Asia = df.loc[df['Continent'] == 'Asia']

Europe = df.loc[df['Continent'] == 'Europe']

Africa = df.loc[df['Continent'] == 'Africa']

North_America = df.loc[df['Continent'] == 'North America']

South_America = df.loc[df['Continent'] == 'South America']



trace0 = go.Box(

    y=Asia['Life Ladder'],

    name = 'Asia',

    boxpoints='all',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=Europe['Life Ladder'],

    name = 'Europe',

    boxpoints='all',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

trace2 = go.Box(

    y=Africa['Life Ladder'],

    name = 'Africa',

    boxpoints='all',

    marker = dict(

        color = 'rgba(12, 50, 196, 0.6)',

    )    

)

trace3 = go.Box(

    y=North_America['Life Ladder'],

    name = 'North America',

    boxpoints='all',

    marker = dict(

        color = 'rgba(171, 50, 96, 0.6)',

    )

)    

trace4 = go.Box(

    y=South_America['Life Ladder'],

    name = 'South America',

    boxpoints='all',

    marker = dict(

        color = 'rgba(80, 26, 80, 0.8)',

    )

)

data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(title='')

fig = go.Figure(data=data, layout=layout)

iplot(fig)