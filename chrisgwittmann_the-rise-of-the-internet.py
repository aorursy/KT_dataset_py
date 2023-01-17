import numpy as np

import pandas as pd



import plotly.offline as py

py.init_notebook_mode() 
import sqlite3 



conn = sqlite3.connect(r'../input/database.sqlite')

c = conn.cursor()
columns = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode', 'Year', 'Value']

internet_query = c.execute("""SELECT * 

                              FROM Indicators 

                              WHERE IndicatorCode = 'IT.NET.USER.P2'

                              """)



internet = pd.DataFrame([row for row in internet_query], columns=columns)

internet.head()
population_query = c.execute("""SELECT * 

                                FROM Indicators 

                                WHERE IndicatorCode = 'SP.POP.TOTL'

                                """)



population = pd.DataFrame([row for row in population_query], columns=columns)

population.head()
# merge relevant information from internet, population databases

net = internet[['CountryName', 'CountryCode', 'Year', 'Value']].rename(columns={'Value': 'Percentage'})

pop = population[['CountryName', 'CountryCode', 'Year', 'Value']].rename(columns={'Value': 'Population'})

df = pd.merge(left=net, right=pop, how='inner', on=['CountryName', 'CountryCode', 'Year'])



# cleaning up data 

df['Percentage'] = df['Percentage'].apply(lambda x: round(x/100, 3))

df['Total'] = df.Percentage * df.Population

df['Total'] = df['Total'].apply(lambda x: int(x))

df = df.drop('Population', axis=1)



df.head()
# only plot years for which we have data available for greater than 80% of countries

c = .8

n_countries = len(df.CountryName.unique())

years = df.Year.unique()

valid_years = [year for year in years if len(df[df.Year==year].CountryName.unique()) > c*n_countries]
"""

Can't import module colorlover on kaggle for now. Until I figure out a solution,

I just explicitly wrote the dictionary the code commented out below would have

assigned to scales.



# nice sequential colorscales for our choropleth maps

import colorlover as cl

from IPython.display import HTML

HTML(cl.to_html(cl.scales['6']['seq']))

scales = cl.scales['6']['seq']

"""



scales = dict(Greens = ['rgb(237,248,233)', 'rgb(199,233,192)', 'rgb(161,217,155)', 'rgb(116,196,118)', 'rgb(49,163,84)', 'rgb(0,109,44)'],

BuGn = ['rgb(237,248,251)', 'rgb(204,236,230)', 'rgb(153,216,201)', 'rgb(102,194,164)', 'rgb(44,162,95)', 'rgb(0,109,44)'],

Oranges = ['rgb(254,237,222)', 'rgb(253,208,162)', 'rgb(253,174,107)', 'rgb(253,141,60)', 'rgb(230,85,13)', 'rgb(166,54,3)'],

Greys = ['rgb(247,247,247)', 'rgb(217,217,217)', 'rgb(189,189,189)', 'rgb(150,150,150)', 'rgb(99,99,99)', 'rgb(37,37,37)'],

RdPu = ['rgb(254,235,226)', 'rgb(252,197,192)', 'rgb(250,159,181)', 'rgb(247,104,161)', 'rgb(197,27,138)', 'rgb(122,1,119)'],

YlGn = ['rgb(255,255,204)', 'rgb(217,240,163)', 'rgb(173,221,142)', 'rgb(120,198,121)', 'rgb(49,163,84)', 'rgb(0,104,55)'],

PuRd = ['rgb(241,238,246)', 'rgb(212,185,218)', 'rgb(201,148,199)', 'rgb(223,101,176)', 'rgb(221,28,119)', 'rgb(152,0,67)'],

OrRd = ['rgb(254,240,217)', 'rgb(253,212,158)', 'rgb(253,187,132)', 'rgb(252,141,89)', 'rgb(227,74,51)', 'rgb(179,0,0)'],

Blues = ['rgb(239,243,255)', 'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)', 'rgb(49,130,189)', 'rgb(8,81,156)'],

YlOrBr = ['rgb(255,255,212)', 'rgb(254,227,145)', 'rgb(254,196,79)', 'rgb(254,153,41)', 'rgb(217,95,14)', 'rgb(153,52,4)'],

PuBuGn = ['rgb(246,239,247)', 'rgb(208,209,230)', 'rgb(166,189,219)', 'rgb(103,169,207)', 'rgb(28,144,153)', 'rgb(1,108,89)'],

YlGnBu = ['rgb(255,255,204)', 'rgb(199,233,180)', 'rgb(127,205,187)', 'rgb(65,182,196)', 'rgb(44,127,184)', 'rgb(37,52,148)'],

GnBu = ['rgb(240,249,232)', 'rgb(204,235,197)', 'rgb(168,221,181)', 'rgb(123,204,196)', 'rgb(67,162,202)', 'rgb(8,104,172)'],

YlOrRd = ['rgb(255,255,178)', 'rgb(254,217,118)', 'rgb(254,178,76)', 'rgb(253,141,60)', 'rgb(240,59,32)', 'rgb(189,0,38)'],

Purples = ['rgb(242,240,247)', 'rgb(218,218,235)', 'rgb(188,189,220)', 'rgb(158,154,200)', 'rgb(117,107,177)', 'rgb(84,39,143)'],

BuPu = ['rgb(237,248,251)', 'rgb(191,211,230)', 'rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(136,86,167)', 'rgb(129,15,124)'],

PuBu = ['rgb(241,238,246)', 'rgb(208,209,230)', 'rgb(166,189,219)', 'rgb(116,169,207)', 'rgb(43,140,190)', 'rgb(4,90,141)'],

Reds = ['rgb(254,229,217)', 'rgb(252,187,161)', 'rgb(252,146,114)', 'rgb(251,106,74)', 'rgb(222,45,38)', 'rgb(165,15,21)'])
# putting our sequential colorscales in a usable format

percentiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

colorscales = {}

for scale in scales.keys():

    colorscales[scale] = [[p, rgb] for p, rgb in zip(percentiles, scales[scale])]
""" 

All we're doing here is creating the necessary structures to update our plots.

The steps will be used to update the map by year on a sliding selector.

The buttons will be used for updating the colorscale on a drop-down selector.

"""



steps = []

for i, year in enumerate(valid_years):

    # update so only selected year's data is plotted

    visible = [False]*len(valid_years)

    visible[i] = True

    step = dict(

        args=['visible', visible],

        label=year,

        method='restyle'

    )

    steps.append(step)



buttons = []

for colorscale in sorted(colorscales):

    button = dict(args=['colorscale', [colorscales[colorscale]]*len(valid_years)],

                  label=colorscale,

                  method='restyle'

                  )

    buttons.append(button)

    
data = list([dict(

        type = 'choropleth',

        locations = df[df.Year == year]['CountryCode'],

        locationmode = 'ISO-3',

        z = df[df.Year == year]['Percentage'],

        # zmin, zmax account for outliers so as not to skew colormap

        zmin = np.percentile(df.Percentage, 5),

        zmax = np.percentile(df.Percentage, 95),

        text = df[df.Year == year]['CountryName'],

        colorscale = colorscales['Blues'],

        autocolorscale = False,

        # make 1990 the default year for which data is visible

        visible = year == min(valid_years),

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False

            )

        ) 

        for year in valid_years])



    

layout = dict(

    title = 'Internet Use Percentage',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    ),

    sliders = [dict(

        active = max(valid_years),

        currentvalue = {"prefix": "Year: "},

        steps = steps

    )],

    updatemenus = [dict(

            x=-0.05,

            y=1,

            yanchor='top',

            buttons=buttons

    )]

)



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='internet-user-percentage-choropleth')
data = list([dict(

        type = 'choropleth',

        locations = df[df.Year == year]['CountryCode'],

        locationmode = 'ISO-3',

        z = df[df.Year == year]['Total'],

        # zmin, zmax account for outliers so as not to skew colormap

        zmin = np.percentile(df.Total, 5),

        zmax = np.percentile(df.Total, 95),

        text = df[df.Year == year]['CountryName'],

        colorscale = colorscales['Blues'],

        autocolorscale = False,

        # make 1990 the default year for which data is visible

        visible = year == min(valid_years),

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False

            )

        ) 

        for year in valid_years])



layout = dict(

    title = 'Total Internet Users',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    ),

    sliders = [dict(

        active = max(valid_years),

        currentvalue = {"prefix": "Year: "},

        steps = steps

    )],

    updatemenus = [dict(

            x=-0.05,

            y=1,

            yanchor='top',

            buttons=buttons

    )]

)



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='total-internet-users-choropleth')