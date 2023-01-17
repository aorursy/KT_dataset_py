import bq_helper

fars = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                dataset_name="nhtsa_traffic_fatalities")
fars.head("accident_2016")
#Get data of fatalities, with/without drunk drivers and total fatalities
    #note: "dd" is being designated as "drunk driver" NOT "designated driver"
query_dd =        """
                     SELECT COUNT(number_of_fatalities) AS fatalities_with_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 1
                  """

query_no_dd =     """
                     SELECT count(number_of_fatalities) AS fatalities_no_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 0
                  """

query_all_fatal = """
                     SELECT count(number_of_fatalities) AS total_fatalities
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  """

#run and print the queries
fatal_dd = fars.query_to_pandas_safe(query_dd)
print(fatal_dd)

print("_"*50)

fatal_no_dd = fars.query_to_pandas_safe(query_no_dd)
print(fatal_no_dd)

print("_"*50)

all_fatal = fars.query_to_pandas_safe(query_all_fatal)
print(all_fatal)

#check for null values in fatalities column
query_fatal_null = """
                      SELECT COUNT(number_of_fatalities) AS number_null_fatalities
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                      WHERE number_of_fatalities IS NULL
                   """

#check for null values in drunk drivers column
query_dd_null = """
                   SELECT COUNT(number_of_drunk_drivers) AS number_null_drunk_drivers
                   FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                   WHERE number_of_drunk_drivers IS NULL
                """

#run queries
null_fatal = fars.query_to_pandas_safe(query_fatal_null)
null_dd = fars.query_to_pandas_safe(query_dd_null)

#print queries
print(null_fatal)
print("_"*50)
print(null_dd)

#check for accident records with no drunk drivers and no fatalities
query_neither = """
                           SELECT COUNT(number_of_fatalities) AS nonFatal_noDrunkDriver
                           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                           WHERE number_of_fatalities = 0 AND number_of_drunk_drivers = 0
                        """
neither = fars.query_to_pandas_safe(query_neither)
print(neither)
#check for accidents with more than one drunk driver
query_dd =        """
                     SELECT COUNT(number_of_fatalities) AS fatalities_with_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers >= 1
                  """
##next two are same queries as the first set we ran^^^
query_no_dd =     """
                     SELECT count(number_of_fatalities) AS fatalities_no_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 0
                  """

query_all_fatal = """
                     SELECT count(number_of_fatalities) AS total_fatalities
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  """

#run and print the queries
fatal_dd = fars.query_to_pandas_safe(query_dd)
print(fatal_dd)

print("_"*50)

fatal_no_dd = fars.query_to_pandas_safe(query_no_dd)
print(fatal_no_dd)

print("_"*50)

all_fatal = fars.query_to_pandas_safe(query_all_fatal)
print(all_fatal)
#print the first few columns of the drimpair_2016 table
fars.head("drimpair_2016")
#Count number of rows in accident table
query_check_accident = """
                          SELECT COUNT(*) AS num_rows
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                       """

#Count number of distinct accident identifiers (consecutive_number) in accident table
query_distinct_accident = """
                             SELECT COUNT(DISTINCT consecutive_number) AS num_distinct_rows
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                          """

#run queries
check_accident = fars.query_to_pandas_safe(query_check_accident)
check_distinct_accident = fars.query_to_pandas_safe(query_distinct_accident)

#print
print(check_accident)
print("_"*50)
print(check_distinct_accident)
#Count number of rows in drimpair table
query_check_accident = """
                          SELECT COUNT(*) AS num_rows
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016`
                       """
#Count number of distinct identifiers in drimpair table
query_distinct_drimpair = """
                             SELECT COUNT(DISTINCT consecutive_number) AS num_distinct_rows
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016`
                          """

#run queries
check_accident = fars.query_to_pandas_safe(query_check_accident)
check_distinct_drimpair = fars.query_to_pandas_safe(query_distinct_drimpair)

#print
print(check_accident)
print("_"*50)
print(check_distinct_drimpair)
query_how_drunk = """
                     WITH fatal_drunk AS
                     (
                         SELECT DISTINCT consecutive_number AS match
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                         WHERE number_of_drunk_drivers >= 1
                     ) --grab identifiers with one or more drunk drivers
                     
                     SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                            COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` AS drimp
                     JOIN fatal_drunk ON
                         fatal_drunk.match = drimp.consecutive_number
                     GROUP BY driver_condition
                     ORDER BY count_ DESC
                  """
#run query
how_drunk = fars.query_to_pandas_safe(query_how_drunk)

#print
print(how_drunk)
#CTE part of last query^
query_1stHalf = """
                           SELECT DISTINCT consecutive_number AS num_distinct
                           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                           WHERE number_of_drunk_drivers >= 1
                       """
#run
check_1stHalf = fars.query_to_pandas_safe(query_1stHalf)

#print
print(check_1stHalf)
#non CTE part of query^^
query_2ndHalf = """
                     SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                            COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` /*AS drimp
                     INNER JOIN acc ON acc.num_distinct = drimp.consecutive_number*/ --commenting these
                     GROUP BY driver_condition
                     ORDER BY count_ DESC
                  """
#run
check_2ndHalf = fars.query_to_pandas_safe(query_2ndHalf)

#print
print(check_2ndHalf)
#attempting original query^^^ with different clause/statements
query_draccident = """
                      SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                             COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` AS drimp
                      WHERE drimp.consecutive_number IN 
                      (
                          SELECT DISTINCT consecutive_number
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS acc
                          WHERE acc.number_of_drunk_drivers >= 1
                      )--Instead of using CTE we are trying the WHERE IN clause
                      GROUP BY driver_condition
                      ORDER BY count_ DESC
                   """

#run
draccident = fars.query_to_pandas_safe(query_draccident)

#print
print(draccident)
import matplotlib.pyplot as plt

plt.bar(draccident.driver_condition, draccident.count_)
plt.xticks(rotation=90)
plt.title("Driver Impairments in Drunk Driving Accidents")
import seaborn as sns

#set up plot
ax=plt.axes()
draccident_plot = sns.barplot(x="driver_condition", y="count_", data=draccident, palette="hls", ax=ax)
ax.set_title("Driver Impairments in Drunk Driving Accidents")
#rotate x-axis labels
for tick in draccident_plot.get_xticklabels():
    tick.set_rotation(90)
#query to grab state (name from accident_2016 table), 
#obstruction, and count of obstructions from vision_2016 table
query_vision = """
                  WITH state AS
                  (
                      SELECT state_number AS s_num, state_name
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  )
                  SELECT drivers_vision_obscured_by_name AS obstruction, state.state_name,
                         COUNT(drivers_vision_obscured_by_name) as obstruction_count
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                  JOIN state ON vision.state_number = state.s_num
                  GROUP BY state_name, obstruction
                  ORDER BY obstruction_count DESC
               """

#run query
vision_obstruct = fars.query_to_pandas_safe(query_vision)

#print
print(vision_obstruct)
#set up plot
obstruction_plot = sns.barplot(x="obstruction", y="obstruction_count", 
                               data=vision_obstruct, hue="state_name")

#move legend outside of figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#rotate x-axis labels
for tick in obstruction_plot.get_xticklabels():
    tick.set_rotation(90)
#query to grab obstruction data from Texas
query_texas_vis = """
                      SELECT drivers_vision_obscured_by_name AS obstruction,
                             COUNT(drivers_vision_obscured_by_name) as obstruction_count
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                      WHERE vision.state_number = 48
                      GROUP BY obstruction
                      ORDER BY obstruction_count DESC
                  """

#run query
texas_vis = fars.query_to_pandas_safe(query_texas_vis)

#plot
ax = plt.axes()
texas_vis_plot = sns.barplot(x="obstruction", y="obstruction_count", data=texas_vis, ax = ax)
ax.set_title("Obstructions in Texas")

#rotate x-axis labels
for tick in texas_vis_plot.get_xticklabels():
    tick.set_rotation(90)
number = 1
null_states = [3,7,14,43,52] #these numbers are missing or either Puerto Rico/Virgin Islands
                             #handy-dandy manual^ :)
#set up arrays
numbers = []
trash = []

#for loop to store numbers that indicate states
for number in range(55):
    number += 1
    if number in null_states:
        trash.append(number)
    else:
        numbers.append(number)

#quick check
#print(numbers)
#query to grab obstruction data from Texas
for i in numbers:
    i_allstates_vis =     """
                             SELECT drivers_vision_obscured_by_name AS obstruction,
                                    COUNT(drivers_vision_obscured_by_name) as obstruction_count
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                             WHERE vision.state_number = (?)
                             GROUP BY obstruction
                             ORDER BY obstruction_count DESC
                          """, (i)

#run query
allstates_vis = fars.query_to_pandas_safe(i_allstates_vis)