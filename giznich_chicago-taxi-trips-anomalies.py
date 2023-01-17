import numpy as np
import pandas as pd
import os
import bq_helper
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
%matplotlib inline
# set everything to query the taxi_trips table:
chicago_taxi = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_taxi_trips")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_taxi_trips")
# list tables
bq_assistant.list_tables()

# display 3 columns
bq_assistant.head("taxi_trips", num_rows=3)
# output the column names
bq_assistant.head("taxi_trips", num_rows=3).columns
# we will be looking only at these columns
columns_of_interest = ['trip_seconds', 'trip_miles', 'fare']
columns_for_query = ','.join(columns_of_interest)
# prepairing a query to extract columns of interest for 2016:
query_2016 = """SELECT %s
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
   EXTRACT(YEAR FROM trip_start_timestamp)=2016
        """ %(columns_for_query)
# let's estiamate the query size (only about 3 gigs)
chicago_taxi.estimate_query_size(query_2016)
# save the query results into a dataframe
df_2016 = chicago_taxi.query_to_pandas_safe(query_2016, max_gb_scanned=30)
# how many rows in the dataframe?
len(df_2016)
(df_2016==0.).sum().div(len(df_2016)/100.).plot(kind='bar', title='Percantage of zero-value records for 2016', figsize=(10, 5))
# boolean indexing which determines whether trip_seconds and trip_miles greater than zero:
is_seconds_and_miles_gtz = (df_2016.trip_seconds > 0.) & (df_2016.trip_miles > 0.)
# boolean indexing which determines whether fare and trip_miles greater than zero:
is_fare_and_miles_gtz = (df_2016.fare > 0.) & (df_2016.trip_miles > 0.)
# adding average speed in MPH into the data frame:
df_2016['aver_speed_mph'] = np.nan
df_2016.loc[is_seconds_and_miles_gtz,'aver_speed_mph'] = 3600. * df_2016[is_seconds_and_miles_gtz].trip_miles.div(df_2016[is_seconds_and_miles_gtz].trip_seconds)
df_2016['fare_per_mile'] = np.nan
df_2016.loc[is_fare_and_miles_gtz, 'fare_per_mile'] = df_2016[is_fare_and_miles_gtz].fare.div(df_2016[is_fare_and_miles_gtz].trip_miles)
def plot_loglog_hist(x, n_bins=100, title='', xlabel=''):
    """
    Input:
    x - parameter of interest in the form of pandas series
    n_bins - number of bins in log scale
    title - title for the plot
    xlabel - label for the x-axis
    Output:
    histogram in loglog scale with log-space bins
    """
    x_min, x_max = x.min(), x.max()
    if x_min == 0.:
        # add small value to zero to avoid an error while taking log in the furter steps
        x_min += 0.001
    # bins in a log space:    
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
    _, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    x.plot(ax=ax, kind='hist', logx=True, logy=True, bins=log_bins, figsize=(20, 5),
          title=title)
plot_loglog_hist(df_2016.trip_seconds.div(3600.), title='Distribution of Trip Duration', xlabel='Trip duration, hours')
plot_loglog_hist(df_2016.trip_miles, title='Distribution of Trip Distance', xlabel='Trip miles')
plot_loglog_hist(df_2016[df_2016.trip_miles > 500.].trip_seconds.div(3600.), title='Distribution of Trip Duration for Trips with distance greater than 500 miles',
                xlabel='Trip duration, hours')
plot_loglog_hist(df_2016.aver_speed_mph, title='Distribution of average speed', xlabel='Average speed, MPH')
# how many trips reached the escape velocity
len(df_2016[df_2016.aver_speed_mph > 25020])
plot_loglog_hist(df_2016.fare_per_mile, title='Distribution of Fare per mile', xlabel='$/mile')
# how many trips costed more than 100k per mile:
len(df_2016[df_2016.fare_per_mile > 1.e5])