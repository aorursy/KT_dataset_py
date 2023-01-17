# SQL date format = YYYY-MM-DD

import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")
# accidents.list_tables()
# query to find out the number of accidents which
# happen on each day of the week

query = """SELECT COUNT(consecutive_number),
                  EXTRACT(DAYOFWEEK from timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
        """

# Now that our query is ready lets run it (safely!) and store 
# the results in a dataframe
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head(n=20)


# Let's plot out data to make sure it has been sorted.

import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

# Which hours of the day do the most accidents occur during?
# Return a table that has information on how many accidents
# occurred in each hour of the day in 2015, sorted by the the
# number of accidents which occurred each hour. Use either the
# accident_2015 or accident_2016 table for this, and the
# timestamp_of_crash column. (Yes, there is an hour_of_crash column,
# but if you use that one you won't get a chance to practice with dates. :P)

query = """SELECT COUNT(consecutive_number) as crash_count,
                  EXTRACT(HOUR from timestamp_of_crash) as hour_of_day
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY hour_of_day
           ORDER BY crash_count DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour.head(n=900)
y = accidents_by_hour.crash_count
x = accidents_by_hour.hour_of_day
accidents_by_hour

def day_to_int(day_str):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if day_str in days:
        return days.index(day_str)
    else:
        raise Exception("day_to_int conversion failed: " + day_str + " is not a valid day " +
                        "please specifiy one of " + ",".join(days))
        
def get_crashes_for_day_of_week(day_int_or_str, year_str='2015'):
    day_int = None
    # convert to int if str was passed
    if type(day_int_or_str) is str:
        day_int = day_to_int(day_int_or_str)
    else:
        day_int = day_int_or_str

    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    """ returns the number of crashes at each hour of the day
        for the specified day of week. Averaged over the year.

        day_int_or_str is from 0 to 6 if int or string such as Tuesday
        0 is monday, 6 is sunday
    """
    day_int = int(day_int)
    day_int = 1 + ((day_int + 1) % 7) # convert so Sunday = 1 (how sql works)
        
    query = """SELECT COUNT(consecutive_number) as crash_count,
                  EXTRACT(HOUR from timestamp_of_crash) as hours_of_day
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_{}`
           WHERE EXTRACT(DAYOFWEEK from timestamp_of_crash) = {}
           GROUP BY hours_of_day
           ORDER BY hours_of_day ASC
        """.format(year_str, int(day_int))   
    
    accidents_by_hour = accidents.query_to_pandas_safe(query)
    crash_counts = accidents_by_hour['crash_count'].as_matrix()
    hours_of_day = accidents_by_hour['hours_of_day'].as_matrix()
    return crash_counts

crash_counts = get_crashes_for_day_of_week("Monday")
crash_counts
import matplotlib.pyplot as plt
import numpy as np

def plot_hourly_crashes_for_days(year):
    year_str = str(year) # make sure year is a string
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    hours = list(range(0, 25, 1))
    plt.xticks(hours)
    plt.xlim(0, 24)
    plt.grid(True)
    plt.title('Average hourly crashes for each day of the week in ' + year_str)
    plt.xlabel('Hour')
    plt.ylabel('Average Crash count')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    crash_counts_for_days = {}
    crash_counts = np.zeros(24)
    for day in days:
        print('getting crash counts for', day)
        crash_counts_for_days[day] = get_crashes_for_day_of_week(day, year_str)

    for day in days:
        crashes_to_plot = list(crash_counts_for_days[day])
        day_index = days.index(day)
        next_day_index = (day_index + 1) % len(days)
        prev_day_index = (day_index - 1) % len(days)
        next_day = days[next_day_index]
        prev_day = days[prev_day_index]
        prev_points = list(crash_counts_for_days[prev_day])
        next_points = list(crash_counts_for_days[next_day])
        cur_points = list(crash_counts_for_days[day])
        # add the first hour from the next day and last hour from the 
        # previous day so the graph line transitions properly
        crashes_to_plot = prev_points[-1:] + cur_points + next_points[:1]
        hour_center = [h - 0.5 for h in range(26)]
        plt.plot(hour_center, crashes_to_plot, label=day)
    plt.legend()

        
plot_hourly_crashes_for_days(2015)
plot_hourly_crashes_for_days(2016)
query = """SELECT COUNT(hit_and_run) as hit_and_run_count,
           registration_state_name
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run = 'Yes'
           GROUP BY registration_state_name
           ORDER BY hit_and_run_count DESC
        """
# vehicle_identification_number_vin
# registration_state_name

#            ORDER BY crash_count DESC
#accidents_by_hour = accidents.query_to_pandas_safe(query)
hit_count = accidents.query_to_pandas_safe(query)

hit_count.head()
#accidents.head('vehicle_2015')
query = """SELECT bus_use,
           registration_state_name
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE bus_use not in ('Not a Bus', 'Unknown')
        """
#            ORDER BY crash_count DESC
#accidents_by_hour = accidents.query_to_pandas_safe(query)
busy = accidents.query_to_pandas_safe(query)
busy
