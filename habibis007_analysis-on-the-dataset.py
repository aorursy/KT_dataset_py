# import the bg helper library for easy querying of big data
import bq_helper
# import the dataset
eclipse_megamovies = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "eclipse_megamovie")
# shows the list of tables available in the dataset
eclipse_megamovies.list_tables()
# view the schema of the specified table
eclipse_megamovies.table_schema("photos_v_0_1")
eclipse_megamovies.head("photos_v_0_1")
# How many totalities were captured
query = """
SELECT (SELECT COUNT(id) 
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_1`
WHERE Totality = True)

+
(SELECT COUNT(id) 
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_2`
WHERE Totality = True)
+
(SELECT COUNT(id) 
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_3`
WHERE Totality = True)

"""
totality = eclipse_megamovies.query_to_pandas_safe(query)
totality.f0_

# 25713 totalities were captured 

"""
Let get users with the model, make, uploaded_date, camera_datetime, is_mobile.

From there, we will check how many pictures were uploaded after the eclipse
"""

query = """
SELECT x.user, x.model, x.make, EXTRACT(DAY FROM x.uploaded_date) as day, x.camera_datetime, x.is_mobile
FROM(SELECT user, model, make, uploaded_date, camera_datetime, is_mobile
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_1`

UNION ALL
SELECT user, model, make, uploaded_date, camera_datetime, is_mobile
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_2`

UNION ALL

SELECT user, model, make, uploaded_date, camera_datetime, is_mobile
FROM `bigquery-public-data.eclipse_megamovie.photos_v_0_3`) x



"""
data = eclipse_megamovies.query_to_pandas_safe(query)

#view sample
data.sample(10)
# pictures uploaded after the day of the eclipse
data[data['day'] > 21].sample(20)



