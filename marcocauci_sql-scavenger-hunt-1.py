import pandas as pd
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality", num_rows = 10)
unit_query = """
        SELECT DISTINCT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm"
        """
open_aq.estimate_query_size(unit_query)
no_ppm_unit = open_aq.query_to_pandas(unit_query)
no_ppm_unit.head()
no_ppm_unit.count()
poll_0_query = """
        SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
open_aq.estimate_query_size(poll_0_query)
poll_0 = open_aq.query_to_pandas(poll_0_query)
poll_0.head()
poll_0.pollutant.value_counts()
no_ppm_unit.to_csv("no_ppm_countries.csv")
poll_0.to_csv("pollutants0.csv")