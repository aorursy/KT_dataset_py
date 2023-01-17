import pandas

import os

os.chdir("/kaggle/working/")
from google.cloud import bigquery

client = bigquery.Client()

ds_ref = client.dataset('chicago_taxi_trips', project='bigquery-public-data')

ds = client.get_dataset(ds_ref)

tbl = client.get_table(ds.table('taxi_trips'))
# Number Trips with 0 trip_miles is 40,230,324 in the filtered data

# Number of Trips with 0 trip_second is 10,087,103 in the filtered data



sql = """

with business_filters as 

(SELECT trip_start_timestamp,

extract(dayofweek from cast(trip_start_timestamp as date)) day_of_week,

     extract(month from cast(trip_start_timestamp as date)) month_of_year,

     if( ((time_diff(time(trip_start_timestamp),time "06:0:00",hour)>0

     and time_diff(time(trip_start_timestamp),time "06:00:00",hour)<=4.5)

     or (time_diff(time(trip_start_timestamp),time "06:00:00",hour)>9

     and time_diff(time(trip_start_timestamp),time "06:00:00",hour)<=13.5)),1,0) as peak_hours_flag,

     if( (time_diff(time(trip_start_timestamp),time "06:0:00",hour)>4.5

     and time_diff(time(trip_start_timestamp),time "06:00:00",hour)<=9),1,0) as day_hours_flag,

     if( time_diff(time(trip_start_timestamp),time "06:0:00",hour)>13.5,1,0) as night_hours_flag,

trip_end_timestamp,

trip_seconds,

trip_miles,

pickup_census_tract,

dropoff_census_tract,

pickup_community_area,

dropoff_community_area,

fare,

trip_total,

tolls,

tips,

extras,

payment_type,

ifnull(company,"Unspecified") as company,

pickup_latitude,

pickup_longitude,

pickup_location,

dropoff_latitude,

dropoff_longitude,

dropoff_location

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE trip_start_timestamp is not null

and (trip_end_timestamp is not null or trip_seconds>10)

and trip_start_timestamp!=trip_end_timestamp

and pickup_census_tract is not null

and dropoff_census_tract is not null

and pickup_location is not null

and dropoff_location is not null

and trip_miles>0

and pickup_community_area is not null

and dropoff_community_area is not null

)



, speed_filter as 

(select * from business_filters

where trip_miles/(trip_seconds/3600)<=70)



select day_of_week,

month_of_year,

peak_hours_flag,

day_hours_flag,

night_hours_flag,

pickup_census_tract,

dropoff_census_tract,

payment_type,

pickup_community_area,

dropoff_community_area,

avg(tolls) as tolls,

avg(fare) as fare,

avg(tips) as tips,

avg(extras) as extras,

avg(trip_total) as trip_total,

avg(trip_miles) as trip_miles,

avg(trip_seconds) as trip_seconds

from speed_filter

group by day_of_week,

month_of_year,

peak_hours_flag,

day_hours_flag,

night_hours_flag,

pickup_census_tract,

dropoff_census_tract,

pickup_community_area,

dropoff_community_area,

payment_type









LIMIT 500000000

"""

raw_aggregated_filter = client.query(sql).to_dataframe()

raw_aggregated_filter.to_csv('raw_aggregated_filteredCompany.csv',index=False)