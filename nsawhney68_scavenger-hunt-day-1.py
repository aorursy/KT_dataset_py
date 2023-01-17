import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

QUERY = """
        SELECT distinct country,pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm" 
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
bq_assistant.estimate_query_size(QUERY)

df_not_ppm = bq_assistant.query_to_pandas(QUERY)
df_not_ppm.to_csv("not_ppm_countries.csv")

df_not_ppm.all

QUERY = """
        SELECT distinct country
        FROM `bigquery-public-data.openaq.global_air_quality`
       
        """
df = bq_assistant.query_to_pandas(QUERY)
df.head(100)
QUERY = """
        SELECT distinct country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = "ppm" 
        """
df_ppm = bq_assistant.query_to_pandas(QUERY)
df_ppm.all
df_ppm.to_csv("ppm_countries.csv")
QUERY = """
        SELECT pollutant, avg(value),min(value),max(value)
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
        """
bq_assistant.estimate_query_size(QUERY)
df_zerovalue = bq_assistant.query_to_pandas(QUERY)
df_zerovalue.all
QUERY = """
        SELECT distinct pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        where value = 0
        """
df_zerovalue = bq_assistant.query_to_pandas(QUERY)
df_zerovalue.all
df_zerovalue.to_csv("zero_valued_pollutants.csv")
QUERY = """
        SELECT distinct averaged_over_in_hours
        FROM `bigquery-public-data.openaq.global_air_quality`
      
        """
df = bq_assistant.query_to_pandas(QUERY)
df.all
QUERY = """
        SELECT country, pollutant, avg(value),min(value),max(value)
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY country, pollutant
        HAVING avg(value) = 0
        """
bq_assistant.estimate_query_size(QUERY)
df_zerovalue = bq_assistant.query_to_pandas(QUERY)
df_zerovalue.all
df_zerovalue.to_csv("zero_valued_pollutants_for_country.csv")
import plotly
import plotly.plotly as py
QUERY = """
        SELECT location, city, country, latitude, longitude, avg(value) as avg_value, min(timestamp),max(timestamp),count(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE extract(YEAR FROM timestamp) = 2018 and 
        extract(MONTH from timestamp) = 1
        and pollutant = 'o3' and unit != "ppm"-- and extract(HOUR from timestamp) BETWEEN 9 and 18
        GROUP BY location, city,country, latitude, longitude
        ORDER BY avg_value desc
        """
df = bq_assistant.query_to_pandas(QUERY)
df.all
scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

data = [ dict(
    lat = df['latitude'],
    lon = df['longitude'],
    text = df['avg_value'].astype(str) + ' ug/mcube',
    marker = dict(
        color = df['avg_value'],
        colorscale = scl,
        reversescale = True,
        opacity = 0.7,
        size = 2,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            ticks = "outside",
            ticklen = 3,
            showticksuffix = "last",
            ticksuffix = " inches",
            dtick = 0.1
        ),
    ),
    type = 'scattergeo'
) ]

layout = dict(
    geo = dict(
        scope = 'world',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation = dict(
                lon = -100
            )
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title = 'Average O3 levels in Jan 2018',
)
fig = { 'data':data, 'layout':layout }
plotly.tools.set_credentials_file(username='nsawhney', api_key='TTmeJTvzDcFVOmyva0Is')
py.plot(fig)
import matplotlib.pyplot as plt
plt.scatter(df.latitude, df.longitude, c=df.avg_value)
plt.colorbar()
plt.show()
QUERY = """
        SELECT location, city, extract(YEAR FROM timestamp), extract(MONTH from timestamp) , avg(value) as avg_value, 
        unit,timestamp,count(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE city = 'Delhi'
        and pollutant = "o3" --and unit != "ppm" --and extract(HOUR from timestamp) BETWEEN 9 and 21
        GROUP BY location, city,extract(YEAR FROM timestamp), extract(MONTH from timestamp),unit,timestamp
        ORDER BY avg_value desc
        """
df = bq_assistant.query_to_pandas(QUERY)
df.all

QUERY = """
        SELECT city, country, avg(value) as avg_value, 
        min(timestamp),max(timestamp),count(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE --extract(YEAR FROM timestamp) = 2018 and 
        --extract(MONTH from timestamp) = 1
        pollutant = 'o3' and unit != "ppm" and extract(HOUR from timestamp) BETWEEN 9 and 21
        GROUP BY  city,country
        HAVING count(city) >= 20
        ORDER BY avg_value desc
        """
df = bq_assistant.query_to_pandas(QUERY)
df.all
df.head(1)
QUERY = """
        SELECT location, city, extract(YEAR FROM timestamp), extract(MONTH from timestamp) , avg(value) as avg_value, 
        unit,timestamp,count(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE city = "AIRPARIF"
        and pollutant = "o3" --and unit != "ppm" -and extract(HOUR from timestamp) BETWEEN 9 and 21
        GROUP BY location, city,extract(YEAR FROM timestamp), extract(MONTH from timestamp),unit,timestamp
        ORDER BY avg_value desc
        """
df = bq_assistant.query_to_pandas(QUERY)
df.all