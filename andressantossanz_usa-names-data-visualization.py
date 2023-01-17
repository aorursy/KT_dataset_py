#Data Analysis

import numpy as np 

import pandas as pd 



#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns



from google.cloud import bigquery 

#Declare the Big Query Client

bq = bigquery.Client()
tables_info_query = """

SELECT

    *

FROM

    `bigquery-public-data.usa_names.__TABLES__`

"""



table_info_result = bq.query(tables_info_query).to_dataframe()



print('The dataset size is ' + str(round(table_info_result['size_bytes'].sum() /(2**30),3)) + ' GB with ' +  str(table_info_result['row_count'].sum()) + ' rows.')

print('Number of tables: '+ str(table_info_result['table_id'].count()) + '\nTables name: ')

print(table_info_result['table_id'])
tables_info_query = """

SELECT 

    *

FROM

    `bigquery-public-data.usa_names.usa_1910_current`

LIMIT 100

"""



table_info_result = bq.query(tables_info_query).to_dataframe()



table_info_result.head()
name_by_year_query = """

SELECT 

    year,

    name,

    gender,

    SUM(number) appearances

FROM

    `bigquery-public-data.usa_names.usa_1910_current`

GROUP BY gender, year, name

"""



ranking_name_by_year_query = """

SELECT

    year,

    name,

    gender,

    appearances,

    RANK() OVER (PARTITION BY year, gender ORDER  BY appearances DESC) position



FROM ("""+ name_by_year_query +""")

"""



mp_name_by_year_query = """

SELECT

    year,

    name,

    gender,

    appearances

FROM ("""+ranking_name_by_year_query+""")

WHERE position = 1

ORDER BY year DESC, gender

"""



mp_name_by_year_result = bq.query(mp_name_by_year_query).to_dataframe()



#Most popular names of the last 5 years

mp_name_by_year_result.head(10)
most_popular_name_query = """

SELECT 

    name,

    gender,

    SUM(number) appearances

FROM

    `bigquery-public-data.usa_names.usa_1910_current`

GROUP BY gender, name

"""



ranking_mp_name_query = """

SELECT

    name,

    gender,

    appearances,

    RANK() OVER (PARTITION BY gender ORDER  BY appearances DESC) position

FROM ("""+ most_popular_name_query +""")

"""



the_mp_name_query = """

SELECT

    name,

    gender,

    appearances,

    position

FROM ("""+ranking_mp_name_query+""")

WHERE position <= 3

ORDER BY gender, position

"""



the_mp_name_result = bq.query(the_mp_name_query).to_dataframe()



#Most popular names

the_mp_name_result.head(6)
unique_rnk_name_query = """

SELECT

    name,

    gender,

    appearances,

    RANK() OVER (PARTITION BY gender ORDER  BY appearances) position

FROM ("""+ most_popular_name_query +""")

"""



unique_name_query = """

SELECT

    name,

    gender,

    appearances

FROM ("""+ unique_rnk_name_query +""")

WHERE position = 1

ORDER BY gender, name

"""



unique_name_result = bq.query(unique_name_query).to_dataframe()



# Display the results

female_name = ""

male_name = ""

print('Unique Name List:')

print('Gender: Female. \nNumber of appearances: ' + str(unique_name_result[unique_name_result['gender'] == 'F'].iloc[0]['appearances']) + '\nList:')



for i, name in enumerate(unique_name_result[unique_name_result['gender'] == 'F']['name']):

    female_name =  female_name + " "+ name

print(female_name)

    

print('\nGender: Male. \nNumber of appearances: '+ str(unique_name_result[unique_name_result['gender'] == 'M'].iloc[0]['appearances'])  + '\nList:')



for i, name in enumerate(unique_name_result[unique_name_result['gender'] == 'M']['name']):

    male_name =  male_name + " "+ name

print(male_name)
both_sex_name_query ="""

SELECT

    name

FROM ("""+ most_popular_name_query +""")

GROUP BY name

HAVING COUNT(*) = 2

ORDER BY name

"""



both_sex_name_result = bq.query(both_sex_name_query).to_dataframe()



print('Unisex names: \nNumber of unisex names: ' + str(both_sex_name_result.shape[0]) + '\nList of names: ')



unisex_names = ""



for i, name in enumerate(both_sex_name_result['name']):

    unisex_names = unisex_names + " " + name

print(unisex_names)
limit = 5

min_appearances = 1000



clean_unisex_subquery = """

SELECT

    name,

    appearances

FROM ("""+ most_popular_name_query +""")

WHERE name IN ("""+ both_sex_name_query + """) AND gender =

"""



clean_unisex_query = """

SELECT

    FEMALES.name

FROM ("""+ clean_unisex_subquery + """ 'F') FEMALES

INNER JOIN ("""+ clean_unisex_subquery  + """ 'M') MALES ON FEMALES.name = MALES.name

WHERE FEMALES.appearances/MALES.appearances <= """+ str(limit) +""" AND MALES.appearances/FEMALES.appearances <= """ + str(limit) + """ 

AND FEMALES.appearances > """ + str(min_appearances) + """ AND MALES.appearances > """ + str(min_appearances) + """

ORDER BY FEMALES.name

"""





clean_unisex_result = bq.query(clean_unisex_query).to_dataframe()



print('Unisex names: \nNumber of unisex names: ' + str(clean_unisex_result.shape[0]) + '\nPercentage of reduction: '+ str(round(((both_sex_name_result.shape[0]-clean_unisex_result.shape[0])/both_sex_name_result.shape[0])*100 , 2)) + '% \nList of names: ')



unisex_clean_names = ""



for i, name in enumerate(clean_unisex_result['name']):

    unisex_clean_names = unisex_clean_names + " " + name

print(unisex_clean_names)

uknowns_query = most_popular_name_query + """HAVING name = 'Unknown'"""



uknowns_result = bq.query(uknowns_query).to_dataframe()



print('There are a total number of ' + str(uknowns_result['appearances'].sum()) + ' unknowns')

print('Females: ' + str(uknowns_result[uknowns_result['gender'] == 'F'].iloc[0]['appearances']) )

print('Males: '+ str(uknowns_result[uknowns_result['gender'] == 'M'].iloc[0]['appearances']))
uknowns_by_year_query = """

SELECT 

    year,

    SUM(number) appearances

FROM

    `bigquery-public-data.usa_names.usa_1910_current`

GROUP BY  year, name

HAVING name = 'Unknown'

ORDER BY year

"""



uknowns_by_year_result = bq.query(uknowns_by_year_query).to_dataframe()



sns.set_style("darkgrid")

fig, ax = plt.subplots()

sns.lineplot(x="year", y="appearances", 

             data=uknowns_by_year_result)

plt.title('Unknown appearances by year')

plt.show()