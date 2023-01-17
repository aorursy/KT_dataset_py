# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
table = '`bigquery-public-data.openaq.global_air_quality`'
query = f"""SELECT city
            FROM {table}
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
bqh = open_aq

# print a list of all the tables in the hacker_news dataset
bqh.list_tables()
# print information on all the columns in the table
table = "global_air_quality"
bqh.table_schema(table)
# preview the first couple lines of the table
bqh.head(table)
# preview the first ten entries in the by column of the table
columns = ['unit']
num_rows = 10
bqh.head(table, selected_columns=columns, num_rows=num_rows)
# estimate query size
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
bqh.estimate_query_size(query)

# only run this query if it's less than 100 MB
bqh.query_to_pandas_safe(query, max_gb_scanned=0.1)
query = f"""select distinct country, unit
           from {table}"""
bqh.estimate_query_size(query)
country_units = bqh.query_to_pandas_safe(query, max_gb_scanned=0.1)


country_units.head()
mask = country_units.unit != 'ppm'

non_ppm_countries = country_units[mask].country
#* Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
non_ppm_countries
#* Which pollutants have a value of exactly 0?
query = f"""select distinct pollutant
           from {table}
           where value = 0"""

bqh.estimate_query_size(query)
max_gb_scanned = 0.1
pollutants_0 = bqh.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
pollutants_0.head()
query = f"""
        select pollutant, max(value) max_value,
          min(value) min_value
        from {table}
        group by pollutant"""

bqh.estimate_query_size(query)
max_gb_scanned = 0.1
pollutant_min_max_value = bqh.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
pollutant_min_max_value.head()
"""
[SchemaField('location', 'string', 'NULLABLE', 'Location where data was measured', ()),
 SchemaField('city', 'string', 'NULLABLE', 'City containing location', ()),
 SchemaField('country', 'string', 'NULLABLE', 'Country containing measurement in 2 letter ISO code', ()),
 SchemaField('pollutant', 'string', 'NULLABLE', 'Name of the Pollutant being measured. Allowed values: PM25, PM10, SO2, NO2, O3, CO, BC', ()),
 SchemaField('value', 'float', 'NULLABLE', 'Latest measured value for the pollutant', ()),
 SchemaField('timestamp', 'timestamp', 'NULLABLE', 'The datetime at which the pollutant was measured, in ISO 8601 format', ()),
 SchemaField('unit', 'string', 'NULLABLE', 'The unit the value was measured in coded by UCUM Code', ()),
 SchemaField('source_name', 'string', 'NULLABLE', 'Name of the source of the data', ()),
 SchemaField('latitude', 'float', 'NULLABLE', 'Latitude in decimal degrees. Precision >3 decimal points.', ()),
 SchemaField('longitude', 'float', 'NULLABLE', 'Longitude in decimal degrees. Precision >3 decimal points.', ()),
 SchemaField('averaged_over_in_hours', 'float', 'NULLABLE', 'The number of hours the value was averaged over.', ())]
"""
query = f"""select distinct country, city, location,
              pollutant, timestamp, source_name, latitude, longitude, averaged_over_in_hours, unit, value
           from {table}"""
bqh.estimate_query_size(query)
max_gb_scanned = 0.1
pollutant_readings = bqh.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
mask = pollutant_readings.value == 0
pollutants_0 = pollutant_readings[mask]
pollutants_0.head()