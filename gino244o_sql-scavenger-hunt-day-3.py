import bq_helper
# bigquery-public-data.nhtsa_traffic_fatalities
nhtsa = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='nhtsa_traffic_fatalities'
                                )
nhtsa.list_tables()
# accident_2015
print(nhtsa.table_schema('accident_2015'))
nhtsa.head('accident_2015')
# accident_2016
print(nhtsa.table_schema('accident_2016'))
nhtsa.head('accident_2016')
# vehicle_2015
print(nhtsa.table_schema('vehicle_2015'))
nhtsa.head('vehicle_2015')
# vehicle_2016
print(nhtsa.table_schema('vehicle_2016'))
nhtsa.head('vehicle_2016')
q1 = """
select
    count(*)
    , extract(hour from timestamp_of_crash)
from
    `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
group by
    extract(hour from timestamp_of_crash)
order by
    count(*) desc
"""
nhtsa.estimate_query_size(q1)
accidents = nhtsa.query_to_pandas_safe(q1)
accidents
import matplotlib.pyplot as plt
plt.plot(accidents.f0_)
plt.title("Number of Accidents by Rank of Hour in 2016")
q2 = """
select
    count(*)
    , registration_state_name
from
    `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
where
    hit_and_run = 'Yes'
group by
    registration_state_name
order by
    count(*) desc
"""
nhtsa.estimate_query_size(q2)
hit_and_runs = nhtsa.query_to_pandas_safe(q2)
hit_and_runs