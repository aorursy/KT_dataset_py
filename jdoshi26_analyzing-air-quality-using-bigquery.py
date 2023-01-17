import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
bq_assistant.head("pm25_frm_daily_summary", num_rows=3)


QUERY = """
    SELECT
       city_name, extract(Year from date_local) as year,
        AVG(aqi) as average_aqi,MAX(aqi) as max_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      lower(city_name) in ("new york","boston","chicago","san francisco")
      AND sample_duration = "24 HOUR"
      AND poc = 1
      group by 1,2
    ORDER BY year
        """
df = bq_assistant.query_to_pandas(QUERY)



print("Changing datatype of Year")
print('')
print(df.dtypes)
print('')
df.year = pd.to_datetime(df.year, format='%Y')
print(df.dtypes)
#df.plot(x='day_of_year', y='aqi', style='.')
sns.set(font_scale=1)
fg = sns.FacetGrid(data=df, hue='city_name', aspect=4,size=2,palette="Set1")
fg.map(plt.scatter, 'year', 'average_aqi').add_legend().set_axis_labels('year','Average Aqi').set_titles("Average AQI in Major Cities")
QUERY1 = """
    SELECT
       city_name, state_name,
        AVG(aqi) as average_aqi,MAX(aqi) as max_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
    sample_duration = "24 HOUR"
      AND poc = 1
      group by 1,2
    ORDER BY 3 desc
    limit 100
        """
df1 = bq_assistant.query_to_pandas(QUERY1)
print("Seems like California consistently has the worst air quality index")
df1.head(n=10)
print("""Looking over at Column metrics in the data ,we see that California may actually have more number of stations where measurements are taken hence to get a more accurate number on worst AQI states, we need to normalize it
""")
df1['state_name'].value_counts()
