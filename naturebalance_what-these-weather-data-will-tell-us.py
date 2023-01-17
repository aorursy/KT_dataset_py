import pandas as pd

from google.cloud import bigquery

client = bigquery.Client()

QUERY ="""

SELECT  state_name , avg(aqi) as aqi_2014_2017

FROM  `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` 

group by state_name

order by aqi_2014_2017

        """

query_job = client.query(QUERY)  # API request

rows = list(query_job.result()) # Waits for query to finish





pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



QUERY ="""

SELECT  state_name , avg(aqi) as aqi2014

FROM  `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` 

WHERE EXTRACT(YEAR FROM date_local) =2014

group by state_name

order by aqi2014

        """

query_job = client.query(QUERY)  # API request

rows = list(query_job.result()) # Waits for query to finish



aqi2014=pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



#-----------------

QUERY ="""

SELECT  state_name , avg(aqi) as aqi2015

FROM  `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` 

WHERE EXTRACT(YEAR FROM date_local) =2015

group by state_name

order by aqi2015

        """

query_job = client.query(QUERY)  # API request

rows = list(query_job.result()) # Waits for query to finish



aqi2015=pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



#-----------------

QUERY ="""

SELECT  state_name , avg(aqi) as aqi2016

FROM  `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` 

WHERE EXTRACT(YEAR FROM date_local) =2016

group by state_name

order by aqi2016

        """

query_job = client.query(QUERY)  # API request

rows = list(query_job.result()) # Waits for query to finish



aqi2016=pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



#-----------------

QUERY ="""

SELECT  state_name , avg(aqi) as aqi2017

FROM  `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` 

WHERE EXTRACT(YEAR FROM date_local) =2017

group by state_name

order by aqi2017

        """

query_job = client.query(QUERY)  # API request

rows = list(query_job.result()) # Waits for query to finish



aqi2017=pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



#assemble



temp=pd.merge(aqi2014,aqi2015,on='state_name')

temp1=pd.merge(temp,aqi2016,on='state_name')

temp=pd.merge(temp1,aqi2017,on='state_name')

temp
temp.describe()
temp['change']=temp['aqi2014']-temp['aqi2017']

temp.sort_values(by='change')