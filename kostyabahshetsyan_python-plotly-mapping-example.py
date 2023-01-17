from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()



import pandas as pd

import numpy as np
indicators = pd.read_csv("../input/Indicators.csv")
indicators.IndicatorName.unique()
# Update these to plot different indicators

indicatorName = "Life expectancy at birth, total (years)"

indicatorYear = 2013





filtered = indicators[(indicators.IndicatorName==indicatorName) & (indicators.Year==indicatorYear)]
correction = {"Antigua and Barbuda":"Antigua", "Bahamas, The":"Bahamas", "Brunei Darussalam":"Brunei",

"Cabo Verde":"Cape Verde", "Congo, Dem. Rep.":"Democratic Republic of the Congo", "Congo, Rep.":"Republic of Congo", 

"Cote d'Ivoire":"Ivory Coast", "Egypt, Arab Rep.":"Egypt", "Faeroe Islands":"Faroe Islands", "Gambia, The":"Gambia", 

"Iran, Islamic Rep.":"Iran", "Korea, Dem. Rep.":"North Korea", "Korea, Rep.":"South Korea", "Kyrgyz Republic":"Kyrgyzstan",

"Lao PDR":"Laos", "Macedonia, FYR":"Macedonia", "Micronesia, Fed. Sts.":"Micronesia", "Russian Federation":"Russia",

"Slovak Republic":"Slovakia", "St. Lucia":"Saint Lucia", "St. Martin (French part)":"Saint Martin", 

"St. Vincent and the Grenadines":"Saint Vincent", "Syrian Arab Republic":"Syria", "Trinidad and Tobago":"Trinidad", 

"United Kingdom":"UK", "United States":"USA", "Venezuela, RB":"Venezuela", "Virgin Islands (U.S.)":"Virgin Islands", 

"Yemen, Rep.":"Yemen"}



filtered.replace(correction, inplace=True)
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = filtered.CountryCode.values,

        z = filtered.Value.values,

        text = filtered.CountryName,

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Count")

        ) ]



layout = dict(

        title = '{} in {}'.format(filtered.IndicatorName.unique()[0],filtered.Year.unique()[0]),

        geo = dict(

            scope='world',

            projection=dict( type='Mercator' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )