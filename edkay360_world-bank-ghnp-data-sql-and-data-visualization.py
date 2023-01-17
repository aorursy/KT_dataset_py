import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
from subprocess import check_output
import bq_helper
from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
ghnp = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_health_population")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_health_population")
bq_assistant.list_tables()
bq_assistant.head("health_nutrition_population", num_rows=5)
queryGetIndicators = """
SELECT 
    indicator_name, indicator_code
FROM
    `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE 
    year = 1960
    AND
    country_code = "USA"
            """
queryGetIndicators = ghnp.query_to_pandas_safe(queryGetIndicators)
queryGetIndicators.tail()
queryAverageAgeFirstMarriage = """
SELECT
  country_name, country_code,
  ROUND(AVG(value),2) AS average
FROM
  `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE
  indicator_code = "SP.DYN.SMAM.FE"
  AND year > 2000
GROUP BY
  country_name, country_code
ORDER BY
  average
;
        """
averageAgeFirstMarriage = ghnp.query_to_pandas_safe(queryAverageAgeFirstMarriage)
averageAgeFirstMarriage.head()
gdata = [ {
        'type': 'choropleth',
        'locations': averageAgeFirstMarriage.country_code,
        'z': averageAgeFirstMarriage.average,
        'text': averageAgeFirstMarriage.country_name,
        'autocolorscale': True,
        'reversescale': True,
        'marker':  {
            'line': {
                'color': 'rgb(180,180,180)',
                'width': 0.5
                    } 
                   },
            } ]

layout = {
    'title': 'Average Age of First Marriage of Females Since 2000',
    'geo':{'showframe': False,
        'showcoastlines': False,
        'projection': {'type': 'orthographic'}
          }
}

figure = {'data': gdata, 'layout': layout}
iplot(figure)
def plotIndicatorMapOverTime(df,title):
    gDataList = []
    frames = []
    years = df.year.unique().tolist()

    #Create template Figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    figure['layout']['title'] = title
    
    #Define Sliders
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': 400,
                'easing': 'cubic-in-out'
            }
        ],
        'initialValue': str(years[0]),
        'plotlycommand': 'animate',
        'values': years,
        'visible': True
    }

    #Update buttons: Play and Pause
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration':0, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]
    
    sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
    }
    
    minZValue = df.indicator.min()
    maxZValue = df.indicator.max()
    
    for year in years:
        dfYear = df[df.year == year]
        gDataTemp = [ {
            'type': 'choropleth',
            'locations': dfYear.country_code,
            'z': dfYear.indicator,
            'text': dfYear.country_name,
            'autocolorscale': True,
            'marker': {
                'line': {
                    'color':'rgb(180,180,180)',
                    'width':0.5
                        } 
                      },
            'zauto': False,
            'zmin': minZValue,
            'zmax': maxZValue,
        } ]
    
        if year == years[0]:     
            figure['data'] = gDataTemp

    
        frame = {'data': gDataTemp,'name': str(year)}
        figure['frames'].append(frame)
    
        slider_step = {'args': [
            [year],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
             'transition': {'duration': 300}}
         ],
         'label': year,
         'method': 'animate'}
        
        sliders_dict['steps'].append(slider_step)
    
    figure['layout']['sliders'] = [sliders_dict]

    iplot(figure)
    
queryLifeExpectancy = """
SELECT
  country_name, country_code, year, ROUND(value,2) as indicator
FROM
  `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE
  indicator_code = "SP.DYN.LE00.IN"
ORDER BY
  year
;
        """
lifeExpectancy = ghnp.query_to_pandas_safe(queryLifeExpectancy)
plotIndicatorMapOverTime(lifeExpectancy, title = 'Life Expectancy')
queryMortalityRateUnder5 = """
SELECT
  country_name, country_code, year, ROUND(value,2) as indicator
FROM
  `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE
  indicator_code = "SH.DYN.MORT"
ORDER BY
  year
;
        """
mortalityRateUnder5 = ghnp.query_to_pandas_safe(queryMortalityRateUnder5)
plotIndicatorMapOverTime(mortalityRateUnder5, title = 'Mortality Rate under 5 (per 1000)')
queryBirthsPerWoman = """
SELECT
  country_name, country_code, year, ROUND(value,2) as indicator
FROM
  `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE
  indicator_code = "SP.DYN.TFRT.IN"
ORDER BY
  year
;
        """
birthsPerWoman = ghnp.query_to_pandas_safe(queryBirthsPerWoman)
plotIndicatorMapOverTime(birthsPerWoman, title = 'Births Per Woman')
queryBirthRateMinusDeathRate = """
            WITH birthRateTable AS
            (SELECT 
                country_name, country_code, year, ROUND(value,2) AS birthRate 
             FROM 
                 `bigquery-public-data.world_bank_health_population.health_nutrition_population`
             WHERE
                 indicator_code = "SP.DYN.CBRT.IN"
            ),
                deathRateTable AS 
            (SELECT 
                country_name, country_code, year, ROUND(value,2) AS deathRate 
             FROM 
                 `bigquery-public-data.world_bank_health_population.health_nutrition_population`
             WHERE
                 indicator_code = "SP.DYN.CDRT.IN"
            )
            
            SELECT 
                 birthRateTable.country_name, birthRateTable.country_code, birthRateTable.year,
                 birthRateTable.birthRate, deathRateTable.deathRate
            FROM
                 birthRateTable
            INNER JOIN deathRateTable
                 ON birthRateTable.country_code = deathRateTable.country_code
                 AND birthRateTable.year = deathRateTable.year
            ORDER BY
                  birthRateTable.year
 """    

birthRateAndDeathRate = ghnp.query_to_pandas_safe(queryBirthRateMinusDeathRate)
birthRateAndDeathRate['indicator'] = birthRateAndDeathRate['birthRate'] - birthRateAndDeathRate['deathRate']
plotIndicatorMapOverTime(birthRateAndDeathRate, title = 'Birth Rate Minus Death Rate (per 1000 people)')