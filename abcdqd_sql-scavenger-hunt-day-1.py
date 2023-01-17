# Add OpenAQ dataset manually to kernel first
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq 
open_aq = bq.BigQueryHelper(active_project= "bigquery-public-data",
                            dataset_name = "openaq")
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality", num_rows= 10)
# Test the sample query
query = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = "US" """
open_aq.estimate_query_size(query) 
query = """ select distinct unit
            from `bigquery-public-data.openaq.global_air_quality` """
open_aq.query_to_pandas(query)
# Run query and save names of countries to table:
query = """ 
        select distinct country
        from `bigquery-public-data.openaq.global_air_quality`
        where unit != 'ppm'          
        """
countries_query1 = open_aq.query_to_pandas_safe(query).values.tolist()
print(countries_query1)
print('Number of countries = ', len(countries_query1))
# count all countries in dataset
query = """ select count(*) as num_countries
            from (select distinct country
                  from `bigquery-public-data.openaq.global_air_quality`)
        """
open_aq.query_to_pandas(query)
# first create a table that summarises by country and unit
# then summarise this table by count of the number of different units each country (possible values are 1 or 2)
# if a country has 1, it must only use ug/m^3 (as we saw above that each country has used ug/m^3 at least once)
query = """ with summ_1 as
            (select count(value) as num_obs,
                   country,
                   unit
            from `bigquery-public-data.openaq.global_air_quality` 
            group by country, unit
            order by country) 
            select count(num_obs) as num_units_used,
                   country
            from summ_1
            group by country
            having num_units_used = 1
        """
countries_query2 = open_aq.query_to_pandas_safe(query)['country'].values.tolist()

print(countries_query2)
print('Number of countries = ', len(countries_query2))
query = """ select pollutant,
                   count(pollutant) as num_zero_obs
            from `bigquery-public-data.openaq.global_air_quality` 
            where value = 0
            group by pollutant """
open_aq.query_to_pandas(query)
# we create two tables, one that counts the number of observations for each pollutant
# and one which counts the number of zeros for each pollutant
# then we join them at calculate the zero frequency
query = """ with pollutant_summary as
                (select pollutant,
                        count(pollutant) as num_obs
                from `bigquery-public-data.openaq.global_air_quality` 
                group by pollutant),
            pollutant_zeroes as 
                (select pollutant,
                        count(pollutant) as num_zero_obs
                from `bigquery-public-data.openaq.global_air_quality` 
                where value = 0
                group by pollutant)
            select ps.pollutant, 
                   ps.num_obs, 
                   pz.num_zero_obs,
                   num_zero_obs / num_obs * 100 as zero_freq
            from pollutant_summary ps
            left join pollutant_zeroes pz
            on ps.pollutant = pz.pollutant
            order by zero_freq desc """
open_aq.query_to_pandas(query)