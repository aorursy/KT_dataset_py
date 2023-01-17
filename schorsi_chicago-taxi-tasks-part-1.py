#pip install kaggle --upgrade
from google.cloud import bigquery

import numpy as np

import pandas as pd
client = bigquery.Client()

dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")

taxi_dat = client.get_dataset(dataset_ref)
tables = list(client.list_tables(taxi_dat))

for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table('taxi_trips')

table = client.get_table(table_ref)

table.schema
first_query = """SELECT company, COUNT(1) AS number_of_trips 

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY company

ORDER BY number_of_trips DESC

"""





safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

first_query_job = client.query(first_query, job_config=safe_config)

first_query_result = first_query_job.to_dataframe()

display(first_query_result)
plot_dat = first_query_result.head(10)

display(plot_dat)
plot_dat = plot_dat.drop(1)

display(plot_dat)
import matplotlib.pyplot as plt

import seaborn as sns

plt.subplots(figsize=(12, 6))

sns.barplot(x=plot_dat['number_of_trips'],y=plot_dat['company'], palette='Greens_r')#color='#4CB391')
company=plot_dat.company.unique()

print(company)
second_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY year, company

HAVING COUNT(1) >= 10000 /*This creates a much shorter output*/

ORDER BY year, company

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

second_query_job =client.query(second_query,job_config=safe_config)

second_query_result = second_query_job.to_dataframe()

second_query_result.head(10)

years = [2013,2014,2015,2016,2017,2018,2019,2020]

final_frame = pd.DataFrame(index=years,columns=company)

display(final_frame) #Now to populate the data
cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Taxi Affiliation Services'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()

print(cp_query_result)

final_frame['Taxi Affiliation Services'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)

final_frame.head(9)
#cp_query stands for copy pasted query, alot of the following queries were copy pasted off an initial one I wrote

#There probably was a better way to execute this, my guess is using subqueries

cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Flash Cab'

GROUP BY year, company

ORDER BY year

""" 

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()



cp_query_result2015 = pd.DataFrame(data=[[2015, 0, 'Flash Cab']], columns=['year', 'num_trips', 'company'],)

cp_query_result2015.head(10)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])

final_frame['Flash Cab'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)



#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Dispatch Taxi Affiliation'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()





cp_query_result2015 = pd.DataFrame(data=[[2019, 0, 'Dispatch Taxi Affiliation'],[2020, 0, 'Dispatch Taxi Affiliation']], columns=['year', 'num_trips', 'company'],)

cp_query_result2015.head(10)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])



final_frame['Dispatch Taxi Affiliation'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)





#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Yellow Cab'

GROUP BY year, company

ORDER BY year

""" 



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

# API request - run the query, and return a pandas DataFrame

cp_query_result = cp_query_job.to_dataframe()

# View results



cp_query_result2015 = pd.DataFrame(data=[[2020, 0, 'Yellow Cab']], columns=['year', 'num_trips', 'company'],)

cp_query_result2015.head(10)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])



final_frame['Yellow Cab'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)



#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Blue Ribbon Taxi Association Inc.'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()

final_frame['Blue Ribbon Taxi Association Inc.'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)





#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Chicago Carriage Cab Corp'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()



cp_query_result2015 = pd.DataFrame(data=[[2013, 0, 'Chicago Carriage Cab Corp'],[2014, 0, 'Chicago Carriage Cab Corp'],[2015, 0, 'Chicago Carriage Cab Corp']], columns=['year', 'num_trips', 'company'],)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])



final_frame['Chicago Carriage Cab Corp'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)





#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Choice Taxi Association'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()

final_frame['Choice Taxi Association'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)





#####



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'Chicago Elite Cab Corp. (Chicago Carriag'

GROUP BY year, company

ORDER BY year

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()

cp_query_result2015 = pd.DataFrame(data=[[2017, 0, 'Dispatch Taxi Affiliation'],[2018, 0, 'Chicago Elite Cab Corp. (Chicago Carriag'],[2019, 0, 'Dispatch Taxi Affiliation'], [2020, 0, 'Dispatch Taxi Affiliation']], columns=['year', 'num_trips', 'company'],)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])



final_frame['Chicago Elite Cab Corp. (Chicago Carriag'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)



######



cp_query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

COUNT(1) AS num_trips, company

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE company = 'City Service'

GROUP BY year, company

ORDER BY year

""" 





safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

cp_query_job =client.query(cp_query,job_config=safe_config)

cp_query_result = cp_query_job.to_dataframe()



cp_query_result2015 = pd.DataFrame(data=[[2013, 0, 'City Service'],[2014, 0, 'City Service'],[2015, 0, 'City Service']], columns=['year', 'num_trips', 'company'],)

cp_query_result = cp_query_result.append(cp_query_result2015)

cp_query_result = cp_query_result.sort_values(by=['year'])





final_frame['City Service'] = pd.Series([x for x in cp_query_result.num_trips], index=final_frame.index)

final_frame.head(9)
company = list(company)

#print(company)
for col in company:

    sns.lineplot(x=final_frame.index, y=final_frame[col])



#This did not produce what i wanted but it shows an interesting trend

#The blue line represents Taxi Affiliation Services