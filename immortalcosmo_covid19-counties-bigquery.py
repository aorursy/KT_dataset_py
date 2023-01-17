#Generates .csv using bigquery

#Counties in U.S with confirmed cases, deaths ..etc using Google's public data 



from google.cloud import bigquery

import pandas as pd

# Create a "Client" object

client = bigquery.Client()



#Tested on Google BigQuery Sandbox

query = """

SELECT

  date,

  covid19.state,

  covid19.county_name,

  ROUND(confirmed_cases/total_pop *100000,2) AS confirmed_cases_per_100000,

  ROUND(deaths/total_pop *100000,2) AS deaths_per_100000,

  confirmed_cases,

  deaths, 

  total_pop AS county_population,

  county_fips_code

FROM `bigquery-public-data.covid19_usafacts.summary` covid19

JOIN `bigquery-public-data.census_bureau_acs.county_2017_5yr` acs 

ON covid19.county_fips_code = acs.geo_id

WHERE county_fips_code != "00000"

AND confirmed_cases + deaths > 0

ORDER BY date, state, county_name, confirmed_cases_per_100000 DESC, deaths_per_100000 DESC

        """



# Set up the query (limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

df = query_job.to_dataframe()
df['county_fips_code'] = df['county_fips_code'].astype(str)
df.to_csv('/kaggle/working/county.csv',index=False)

#Use this line to include leading Zeros when reading csv

#df = pd.read_csv('county.csv', index_col="date", dtype ={'county_fips_code': 'str'})