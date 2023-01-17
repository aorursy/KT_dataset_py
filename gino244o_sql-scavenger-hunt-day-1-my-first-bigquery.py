import bq_helper
# bigquery-public-data.openaq
openaq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                  dataset_name='openaq'
                                 )
openaq.list_tables()
openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')
query = """
select
    distinct unit
from
    `bigquery-public-data.openaq.global_air_quality`
order by
    unit
"""
openaq.estimate_query_size(query)
units = openaq.query_to_pandas_safe(query)
units.unit
q1 = """
select
    distinct country
from
    `bigquery-public-data.openaq.global_air_quality`
where
    unit != 'ppm'
order by
    country
"""
openaq.estimate_query_size(q1)
countries = openaq.query_to_pandas_safe(q1)
print(countries.country.count(), 'countries')
cols = 10
idx = 0
while True:
    print(countries.country[idx:idx+cols].values)
    idx += cols
    if idx >= countries.country.count():
        break

query = """
select
    distinct pollutant
from
    `bigquery-public-data.openaq.global_air_quality`
order by
    pollutant
"""
openaq.estimate_query_size(query)
allpollutants = openaq.query_to_pandas_safe(query)
allpollutants
query = """
select
    pollutant
    , min(value) min_value
from
    `bigquery-public-data.openaq.global_air_quality`
group by
    pollutant
order by
    pollutant
"""
openaq.estimate_query_size(query)
minpollutants = openaq.query_to_pandas_safe(query)
minpollutants
q2 = """
select
    distinct pollutant
from
    `bigquery-public-data.openaq.global_air_quality`
where
    value = 0
order by
    pollutant
"""
openaq.estimate_query_size(q2)
pollutants = openaq.query_to_pandas_safe(q2)
pollutants