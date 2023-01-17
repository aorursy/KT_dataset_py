import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from wordcloud import WordCloud
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
bq_assistant.list_tables()
query = """SELECT COUNT(*) AS total_rows 
           FROM `bigquery-public-data.openaq.global_air_quality`"""
total_rows = bq_assistant.query_to_pandas_safe(query)
print(total_rows)
query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm'"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
ppm_values.describe()
query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm' AND value >= 0"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
bins = np.linspace(0,400,10,dtype='i')
plt.hist( ppm_values.value, bins=bins,color='green')
ppm_values.describe()
query = """SELECT * 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm' AND value >= 350"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
# save outlier value to filter in next queries
ppm_value_outlier = ppm_values.value[0]
ppm_values
query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
ugm3_values.describe()
query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm' AND value >= 0 AND value < 1000000"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
plt.hist( ugm3_values.value, bins=np.linspace(0,1000000,10))
ugm3_values.describe()
# We can zoom in at values less than 20,000
query = """SELECT value
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm' AND value >= 0 AND value < 20000"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
plt.hist( ugm3_values.value, bins=np.linspace(0,20000,10),log=True)
query = """SELECT longitude, latitude
           FROM `bigquery-public-data.openaq.global_air_quality`
           -- cleaning invalid data
           WHERE value >= 0 AND value < 20000 
           """
df = bq_assistant.query_to_pandas_safe(query)
# Background image got at the internet
# https://vignette.wikia.nocookie.net/pixar/images/b/b0/20100625011514%21World_Map_flat_Mercator.png/revision/latest?cb=20120823094025&format=original
img = mpimg.imread("../input/World_Map_flat_Mercator.png")
plt.figure(figsize=(18,9))
imgplot = plt.imshow(img,zorder=1)
# Have to scale latitudes and longitudes to fit world map at background
plt.scatter((df.longitude+167)*1468/360, (df.latitude*-1+126)*1006/218, c=sns.color_palette("autumn"), s=1, alpha=0.5,zorder=2)
plt.grid(True)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Distribution of Measurement Locations")
plt.show()

# I misnamed the csv file as "cities", sorry, it's "countries"
data = pd.read_csv("../input/OpenAQ_cities.csv", header=0)
country_codes = pd.DataFrame(data,columns=["Country","Code"])
country_codes.set_index('Code');
# Obtain totals per location using normalized values to ppm units
query = """SELECT * FROM (
            WITH norm_values AS 
            (
                SELECT location, city, country, 
                      (value / IF((rtrim(ltrim(unit)) != 'ppm'),1266,1)) as nvalue
                FROM `bigquery-public-data.openaq.global_air_quality`
                -- cleaning invalid data
                WHERE value >= 0 AND value < 20000 
            )
            SELECT location, city, country, SUM(nvalue) AS total_pollution
            FROM norm_values
            GROUP BY location, city, country
            ORDER BY total_pollution DESC
            LIMIT 20)
           ORDER BY total_pollution, country, city, location
            """

df = bq_assistant.query_to_pandas_safe(query)
# Filter that ppm outlier detected previously and get top 10 locations to plot
df = df[df.total_pollution < ppm_value_outlier][:10]
plt.figure(figsize=(12,4))
plt.xlabel("Pollution in ppm")
plt.title("Total pollution: Top 10 Worst Locations")
plt.yticks(np.arange(10),df.location,rotation=0) 
# Using a log scale for better visualization
plt.barh(np.arange(10),df.total_pollution,align='center', tick_label=df.country+'-'+df.city+'-'+df.location,log=False, color=sns.light_palette('purple',10, reverse=False))
plt.show()
# Obtain totals per location using normalized values to ppm units
query = """SELECT * FROM (
            WITH norm_values AS 
            (
                SELECT city, country, 
                      (value / IF((rtrim(ltrim(unit)) != 'ppm'),1266,1)) as nvalue
                FROM `bigquery-public-data.openaq.global_air_quality`
                -- cleaning invalid data
                WHERE value >= 0 AND value < 20000 
            )
            SELECT city, country, SUM(nvalue) AS total_pollution
            FROM norm_values
            GROUP BY city, country
            ORDER BY total_pollution
            LIMIT 100)
            """

df = bq_assistant.query_to_pandas_safe(query)
# Filter that ppm outlier detected previously
df = df[df.total_pollution < ppm_value_outlier]
df[:10]
# Chart countries having best AQ, translate country codes with country names
txt = ''
for n in df.country:
    ctry = country_codes[country_codes['Code'] == n].Country.values[0]
    ctry = ctry.replace( ' ', '')
    txt = txt + ctry + ' '
plt.figure(figsize=(6,6))
wc = WordCloud(background_color='gray', max_font_size=200,
                            width=600,
                            height=400,
                            max_words=25,
                            relative_scaling=.3).generate(txt)
plt.imshow(wc)
plt.title("Countries with more Best AQ Cities", fontsize=14)
plt.axis("off");
query = """SELECT DATETIME_DIFF(now,past,DAY) AS days_update FROM (
            SELECT DATETIME(CURRENT_TIMESTAMP()) AS now, DATETIME(timestamp) AS past           
            FROM `bigquery-public-data.openaq.global_air_quality`
            -- cleaning invalid data
            WHERE value >= 0 AND value < 20000)
           ORDER BY days_update
           """
df = bq_assistant.query_to_pandas_safe(query)
df = df[ df.days_update >= 0]
# save some statistic data for the plot
days_update_mean = df.days_update.mean()
days_update_std = df.days_update.std()
plt.figure(figsize=(12,4))
plt.ylabel("Locations updated")
plt.xlabel("Update delay (days)")
plt.title("Update delay at Locations (days)")
plt.hist( df.days_update, color='y', bins=np.linspace(0,days_update_mean+(2*days_update_std),20));