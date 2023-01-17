# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
dayslist = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
def day_label(dayid):
    return dayslist[dayid]
daycol = [day_label(i-1) for i in  accidents_by_day.f1_]
daycol
accidents_by_day['DayName'] = daycol
print(accidents_by_day)
# Your code goes here :)
query1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
hourslist = ["12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM",
             "6 AM",  "7 AM", "8 AM", "9 AM", "10 AM", "11 AM", 
             "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM",
             "6 PM",  "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"]
def hour_label(hourid):
    return hourslist[hourid]
hourcol = [hour_label(i) for i in  accidents_by_hour.f1_]
accidents_by_hour['HourName'] = hourcol
print(accidents_by_hour)
query2 = """SELECT AllRecs.*,
               case when AccidentHour > 12 
                      then CONCAT(RTRIM(CAST(AccidentHour - 12 AS STRING)), ' PM') 
                    when  AccidentHour = 0
                      then '12 AM'
                      else CONCAT(RTRIM(CAST(AccidentHour AS STRING)), ' AM') 
                    end AS HourName
               FROM
              (SELECT COUNT(consecutive_number) AS AccidentCount, 
                  EXTRACT(HOUR FROM timestamp_of_crash)  AS AccidentHour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ) AllRecs
            ORDER BY AccidentCount DESC
        """
accidents_by_hour_2 = accidents.query_to_pandas_safe(query2)
accidents_by_hour_2
query3 = """SELECT 
            registration_state_name AS State, 
            COUNT(hit_and_run) AS HitRunCount
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """
hitAndRun = accidents.query_to_pandas_safe(query3)
print(hitAndRun)