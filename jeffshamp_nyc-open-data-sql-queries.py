# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Any results you write to the current directory are saved as output.

import bq_helper

import matplotlib.pyplot as plt

ny_data_set = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")
maple_query = """

WITH tree_list AS

(SELECT spc_common

FROM `bigquery-public-data.new_york.tree_census_2015`

INTERSECT DISTINCT

SELECT spc_common

FROM `bigquery-public-data.new_york.tree_census_2015`)

SELECT COUNT(*) AS total_number_tree_species

FROM tree_list 

"""

maples = ny_data_set.query_to_pandas_safe(maple_query)

maples
# MAPLE SYRUP problem - Find the number of red, black, and sugar maple trees in NYC.

maple_query = """

SELECT 

    spc_common AS tree_type,

    COUNT(spc_common) AS total_count

FROM `bigquery-public-data.new_york.tree_census_2015`

WHERE

    status != 'Dead' AND

    spc_common IN ('sugar maple' ,'black maple', 'red maple')

GROUP BY spc_common

"""

maples = ny_data_set.query_to_pandas_safe(maple_query)

maples
maple_query_2 = """

WITH maple_count AS 

    (

    SELECT spc_common AS tree_type,

           COUNT(spc_common) AS count

    FROM   `bigquery-public-data.new_york.tree_census_2015`

    WHERE  status != 'Dead' 

           AND spc_common IN ('sugar maple', 'black maple', 'red maple')

    GROUP BY spc_common

    ) 

SELECT SUM(count) AS total,

CASE

    WHEN tree_type IN ('black maple', 'sugar maple') 

    THEN 'black & sugar maple'

    ELSE 'red maple'

END AS sub_class

FROM maple_count

GROUP BY sub_class

"""

maples_2 = ny_data_set.query_to_pandas_safe(maple_query_2)

maples_2
# Most dangerous borough for pedestrians since 2015 

peds_query = """

WITH totals AS 

(

    SELECT  borough,

            SUM(number_of_pedestrians_killed) AS total_deaths,

            EXTRACT(MONTH FROM timestamp) AS month,

            EXTRACT(YEAR FROM timestamp) AS year

    FROM `bigquery-public-data.new_york.nypd_mv_collisions`

    WHERE borough NOT IN ('')

    GROUP BY borough, year, month

    ORDER BY year, month

),

averages AS 

(

    SELECT  borough, month, year, total_deaths, 

            AVG(total_deaths) over (partition by year) AS yearly_average,

            ROUND( AVG(total_deaths) over (partition by month),2) AS monthly_average

    FROM totals

)

SELECT borough, month,

       year, total_deaths, 

       monthly_average, yearly_average

FROM   averages

WHERE  total_deaths > 3*monthly_average

       AND monthly_average > yearly_average

ORDER BY year, month

"""

ped_death = ny_data_set.query_to_pandas_safe(peds_query)

ped_death
peds_query_2 = """

SELECT  borough,

        SUM(number_of_pedestrians_killed) AS total_deaths

FROM `bigquery-public-data.new_york.nypd_mv_collisions`

GROUP BY borough

"""



ped_death = ny_data_set.query_to_pandas_safe(peds_query_2)

ped_death
peds_query_two = """

SELECT  borough, zip_code, 

        on_street_name, off_street_name, 

        cross_street_name, latitude, longitude

FROM `bigquery-public-data.new_york.nypd_mv_collisions`

WHERE EXTRACT(YEAR FROM timestamp) > 2015 

        AND borough IN ('')

        AND number_of_pedestrians_killed > 0

        LIMIT 20

"""

ped_death_two = ny_data_set.query_to_pandas_safe(peds_query_two)

ped_death_two
#All the noise complaints on Irving Place. 

address_query = """

                 WITH bus_noise AS

                    (

                        SELECT  incident_address,

                                complaint_type 

                        FROM    `bigquery-public-data.new_york.311_service_requests`

                        WHERE   complaint_type like '%Noise%' 

                        

                    )

                 SELECT count(complaint_type) AS count, incident_address AS address

                 FROM bus_noise

                 WHERE incident_address like '%IRVING PLACE%'

                 GROUP BY address

                 HAVING count >= 5

                 ORDER BY count DESC

                """

noise = ny_data_set.query_to_pandas_safe(address_query)

noise