import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from google.cloud import bigquery 

import bq_helper

from bq_helper import BigQueryHelper
#Using BigQuery

bq = bigquery.Client()



#Using BigQueryHelper

google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="data:google_analytics_sample")

bqh = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
#Listing

table_list = bqh.list_tables()



#Print the result. Note: Elements that are not string by string must be parsed in order to be represented. This is done by entering the element in the str() function.

print ('Number of tables: ' + str(len(table_list)) +  '\nFirst 3 tables: ' + str(table_list[0:3]))
bqh.head(table_list[0], num_rows=10)
bqh.table_schema(table_list[0])
#The query to be executed

size_query = """

SELECT

    SUM(size_bytes) Bytes

FROM

    `bigquery-public-data.google_analytics_sample.__TABLES__`

"""



#Query execution and save the result in a data frame. The name of the colums of the dataset are defined in the query.

size_result = bq.query(size_query).to_dataframe()



print('The dataset size is ' + str(round(size_result.iloc[0]['Bytes']/(2**30),3)) + ' GB')
JSON_query = """

SELECT

    totals,

    trafficSource,

    device,

    geoNetwork,

    customDimensions,

    hits

FROM

    `bigquery-public-data.google_analytics_sample."""+ bqh.list_tables()[1] +"""`

LIMIT 1

"""

JSON_result = bq.query(JSON_query).to_dataframe()



print('totals: \n' +

str(JSON_result.iloc[0]['totals']) +

'\n\ntrafficSource: \n'+

str(JSON_result.iloc[0]['trafficSource'])+

'\n\ndevice: \n'+

str(JSON_result.iloc[0]['device']) +

'\n\ngeoNetwork: \n'+

str(JSON_result.iloc[0]['geoNetwork'])+

'\n\ncustomDimensions: \n'+

str(JSON_result.iloc[0]['customDimensions'])+

'\n\nhits: \n' +

str(JSON_result.iloc[0]['hits']))

year_query = """

SELECT substr(date,0,4) Year,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Year

"""

year_result = bq.query(year_query).to_dataframe()

year_result
#The bar plot plot

year_graph = year_result.plot(kind='bar',x='Year',y='Visits')

#Title

year_graph.set_title('Visits per year')

#Removing the legend because does not add any important information

year_graph.legend().remove()

#Set the y and x axis labels

year_graph.set_ylabel('Visits')

year_graph.set_xlabel('Year')

plt.show()
month_query = """

SELECT CONCAT(substr(date,0,4),'-', substr(date,5,2)) Month,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Month

ORDER BY Month

"""

month_result = bq.query(month_query).to_dataframe()

month_result
month_graph = month_result.plot(kind='line',x='Month',y='Visits')

month_graph.set_title('Visits per month')

month_graph.legend().remove()

month_graph.set_ylabel('Visits')

month_graph.set_xlabel('Month')

#Adding a grid in the plot for improve the visualization

month_graph.xaxis.grid(True, linestyle='-.', which='major', color='grey', alpha=.2)

month_graph.yaxis.grid(True, linestyle='-.', which='major', color='grey', alpha=.2)

plt.show()
zone_query = """

SELECT geoNetwork.continent Area,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Area

ORDER BY Visits

"""

zone_result = bq.query(zone_query).to_dataframe()

zone_result
zone_graph = zone_result.plot(kind='bar',x='Area',y='Visits')

zone_graph.set_title('Visits per area')

zone_graph.legend().remove()

zone_graph.set_ylabel('Visits')

zone_graph.set_xlabel('Area')

plt.show()
country_query = """

SELECT geoNetwork.country Country,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Country

ORDER BY Visits DESC

"""

country_result = bq.query(country_query).to_dataframe()

country_result
country_graph = country_result.iloc[0:10].plot(kind='bar',x='Country',y='Visits')

country_graph.set_title('Visits per country')

country_graph.legend().remove()

country_graph.set_ylabel('Visits')

country_graph.set_xlabel('Country')

plt.show()
city_query = """

SELECT geoNetwork.city City,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

WHERE geoNetwork.country = (

    SELECT Country FROM ( 

        SELECT geoNetwork.country Country, COUNT(*) Visits 

            FROM 

                `bigquery-public-data.google_analytics_sample.*`

                GROUP BY Country ORDER BY Visits DESC) S LIMIT 1)

GROUP BY City

ORDER BY Visits DESC

"""

city_result = bq.query(city_query).to_dataframe()

city_result
city_graph = city_result.iloc[1:6].plot(kind='bar',x='City',y='Visits')

city_graph.set_title('Visits per city')

city_graph.legend().remove()

city_graph.set_ylabel('Visits')

city_graph.set_xlabel('City')

plt.show()
city_an_query = """

SELECT geoNetwork.city City,

        geoNetwork.country Country,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

WHERE geoNetwork.country = (

    SELECT Country FROM ( 

        SELECT geoNetwork.country Country, COUNT(*) Visits 

            FROM 

                `bigquery-public-data.google_analytics_sample.*`

                GROUP BY Country ORDER BY Visits DESC) S LIMIT 1)

GROUP BY City, Country

ORDER BY Visits DESC

"""

city_an_result = bq.query(city_an_query).to_dataframe()

city_an_result
city_country_an_query = """

SELECT  City,

        COUNT(*) CountCountry

        FROM (

            SELECT DISTINCT geoNetwork.city City, geoNetwork.country Country  

            FROM `bigquery-public-data.google_analytics_sample.*`

            )

GROUP BY City

HAVING CountCountry > 1

ORDER BY CountCountry DESC         

"""

city_country_an_result = bq.query(city_country_an_query).to_dataframe()



for i in range(city_country_an_result.shape[0]):

    print(str(city_country_an_result.iloc[i]['City']) + ' -> ' + str(city_country_an_result.iloc[i]['CountCountry']))
city_country_test_query = """

    SELECT geoNetwork.city City, 

            geoNetwork.country Country,

            Count(*) Count, 

            ROW_NUMBER() OVER (PARTITION BY geoNetwork.city ORDER BY Count(*) DESC) AS row_num 

    FROM `bigquery-public-data.google_analytics_sample.*`

    GROUP BY Country, City

    ORDER BY City, row_num

"""

city_country_test_result = bq.query(city_country_test_query).to_dataframe()



for i in range(city_country_test_result.shape[0]):

    print(str(city_country_test_result.iloc[i]['City']) + ' -> ' + 

          str(city_country_test_result.iloc[i]['Country']) + 

          ' (RANKING: ' + str(city_country_test_result.iloc[i]['row_num']) + 

          ' COUNTER: ' + str(city_country_test_result.iloc[i]['Count']) + ')' )
city_country_query = """

SELECT 

    Country,

    City

FROM (

        SELECT geoNetwork.city City, 

            geoNetwork.country Country,

            Count(*) Count, 

            ROW_NUMBER() OVER (PARTITION BY geoNetwork.city ORDER BY Count(*) DESC) AS row_num 

        FROM `bigquery-public-data.google_analytics_sample.*`

        GROUP BY Country, City

        ORDER BY City, row_num )

WHERE row_num = 1

ORDER BY Country, City

"""

city_country_result = bq.query(city_country_query).to_dataframe()



for i in range(city_country_result.shape[0]):

    print(str(city_country_result.iloc[i]['Country']) + ' -> ' + str(city_country_result.iloc[i]['City']) )
#We store this query in a variable because it will be used twice in the subsequent query.

country_sub_query = """(

SELECT 

    Country 

FROM ( 

        SELECT geoNetwork.country Country, COUNT(*) Visits 

            FROM 

                `bigquery-public-data.google_analytics_sample.*`

                GROUP BY Country ORDER BY Visits DESC) S 

LIMIT 1)

"""





city_sol_query = """

SELECT geoNetwork.city City,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

WHERE geoNetwork.country = """+ country_sub_query  +""" AND 

        geoNetwork.city  IN (SELECT 

                        City

                    FROM (

                            SELECT geoNetwork.city City, geoNetwork.country Country,Count(*) Contador, 

                            ROW_NUMBER() OVER (PARTITION BY geoNetwork.city ORDER BY Count(*) DESC) AS row_num 

                                FROM `bigquery-public-data.google_analytics_sample.*` GROUP BY Country, City )

            WHERE  row_num = 1 AND Country = """+ country_sub_query  +""")

GROUP BY geoNetwork.city 

ORDER BY Visits DESC

"""

city_sol_result = bq.query(city_sol_query).to_dataframe()



print('Number of records before -> Number of records after\n' 

      + str(city_result.shape[0]) + ' -> ' + str(city_sol_result.shape[0]) 

      + '\nReduction percentage: ' + str(round( 100*(city_result.shape[0]- city_sol_result.shape[0])/city_result.shape[0],2)) + ' %'  )

social_query = """

SELECT socialEngagementType SocialEngadgementType,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY SocialEngadgementType

ORDER BY Visits DESC

"""

social_result = bq.query(social_query).to_dataframe()

social_result
channel_query = """

SELECT channelGrouping ChannelGrouping,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY ChannelGrouping

ORDER BY Visits DESC

"""

channel_result = bq.query(channel_query).to_dataframe()

channel_result
channel_graph = channel_result.plot(kind='bar',x='ChannelGrouping',y='Visits')

channel_graph.set_title('Visits by channel grouping type')

channel_graph.legend().remove()

channel_graph.set_ylabel('Visits')

channel_graph.set_xlabel('Channel grouping type')

plt.show()
channel_month_query = """

SELECT channelGrouping ChannelGrouping,

        CONCAT(substr(date,0,4),'-', substr(date,5,2)) Month,

        COUNT(*) Visits

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY ChannelGrouping, Month

ORDER BY Month, ChannelGrouping

"""

channel_month_result = bq.query(channel_month_query).to_dataframe()

channel_month_result
#To plot multiple lines plot use a for loop

for i in range(channel_result.shape[0]):

      plt.plot(channel_month_result.loc[channel_month_result['ChannelGrouping'] == channel_result.iloc[i]["ChannelGrouping"]]['Month'],channel_month_result.loc[channel_month_result['ChannelGrouping'] == channel_result.iloc[i]["ChannelGrouping"]]['Visits'] , label = channel_result.iloc[i]["ChannelGrouping"])



plt.title('Visits per month by channel grouping type')



plt.ylabel('Visits')

# Rotation of the x axis marks to visualize it properly 

plt.xticks(rotation = '90')

# To improve the display, the legend box is taken out of the drawing area.

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)



plt.grid(True, linestyle='-.', which='major', axis = 'both' ,color='grey', alpha=.2)



#To save the plot

plt.savefig('Visits_perMonth_by_channelGrouping.png')

plt.show()
#Información compuesto por los 3 datos

device_query = """

SELECT device.browser Browser,

        device.operatingSystem OS,

        device.isMobile Smartphone,

        COUNT(*) Count

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Browser, OS, Smartphone

ORDER BY Count DESC

"""

device_result = bq.query(device_query).to_dataframe()



browser_query = """

SELECT device.browser Browser,

        COUNT(*) Count

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY Browser

ORDER BY Count DESC

"""

browser_result = bq.query(browser_query).to_dataframe()



os_query = """

SELECT device.operatingSystem OS,

        COUNT(*) Count

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

GROUP BY OS

ORDER BY Count DESC

"""

os_result = bq.query(os_query).to_dataframe()



browser_desktop_query = """

SELECT device.browser Browser,

        COUNT(*) Count

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

        WHERE device.isMobile = False

GROUP BY Browser

ORDER BY Count DESC

"""

browser_desktop_result = bq.query(browser_desktop_query).to_dataframe()



browser_smartphone_query = """

SELECT device.browser Browser,

        COUNT(*) Count

        FROM 

            `bigquery-public-data.google_analytics_sample.*`

        WHERE device.isMobile = True

GROUP BY Browser

ORDER BY Count DESC

"""

browser_smartphone_result = bq.query(browser_smartphone_query).to_dataframe()
# General data of the charts.

fig, ax = plt.subplots(2,2)

fig.set_figheight(16)

fig.set_figwidth(16)

fig.suptitle('OS and browser information', size = '16')



#Dividers are drawn

plt.plot([-1.75, -1.75], [-1.25, 4.5], color='grey', lw=1, clip_on=False)

plt.plot([-4.25, 1.25], [1.5, 1.5], color='grey', lw=1, clip_on=False)



#Graph 1: Position 0.0

#The DataFrame is modified to be readable, the first 5 results are printed and the rest are grouped under the label "Others"

#First the vectors are assembled

nav_res = []

cont_res = []

cont_sum = 0

for i in range(browser_result.shape[0]):

    if (i <= 4):

        nav_res.append(browser_result.iloc[i]['Browser'])

        cont_res.append(browser_result.iloc[i]['Count'])

    else:

        cont_sum += browser_result.iloc[i]['Count']

            

nav_res.append('Others')

cont_res.append(cont_sum)



#Then the dataset is formed

browser_result_print = pd.DataFrame({"Browser":nav_res, "Count":cont_res})

    

ax[0, 0].pie(browser_result_print.iloc[:]['Count'], labels = browser_result_print.iloc[:]['Browser'], autopct='%.0f%%', startangle=180)

ax[0,0].set_title('Browsers', size = '14')



#Graph 2: Position 0.1

#The same operation is performed with the operating system graphic. 

OS_res = []

cont_res = []

cont_sum = 0

for i in range(os_result.shape[0]):

    if (i <= 5):

        OS_res.append(os_result.iloc[i]['OS'])

        cont_res.append(os_result.iloc[i]['Count'])

    else:

        cont_sum += os_result.iloc[i]['Count']

            

OS_res.append('Others')

cont_res.append(cont_sum)



#Then the dataset is formed

OS_result_print = pd.DataFrame({"OS":OS_res, "Count":cont_res})



ax[0, 1].pie(OS_result_print.iloc[:]['Count'], labels = OS_result_print.iloc[:]['OS'], autopct='%.0f%%' )

ax[0,1].set_title('OS', size = '14')



#Graph 3: Position 1.0

# Use of browsers on desktop computers

nav_res = []

cont_res = []

cont_sum = 0

for i in range(browser_desktop_result.shape[0]):

    if (i <= 4):

        nav_res.append(browser_desktop_result.iloc[i]['Browser'])

        cont_res.append(browser_desktop_result.iloc[i]['Count'])

    else:

        cont_sum += browser_desktop_result.iloc[i]['Count']

            

nav_res.append('Others')

cont_res.append(cont_sum)





browser_result_desktop_print = pd.DataFrame({"Browser":nav_res, "Count":cont_res})

    

ax[1, 0].pie(browser_result_desktop_print.iloc[:]['Count'], labels = browser_result_desktop_print.iloc[:]['Browser'], autopct='%.0f%%', startangle=180 )

ax[1,0].set_title('Desktop browsers', size = '14')



#Graph 4: Position 1.1

# Use of browsers on mobile devices

nav_res = []

cont_res = []

cont_sum = 0

for i in range(browser_smartphone_result.shape[0]):

    if (i <= 4):

        nav_res.append(browser_smartphone_result.iloc[i]['Browser'])

        cont_res.append(browser_smartphone_result.iloc[i]['Count'])

    else:

        cont_sum += browser_smartphone_result.iloc[i]['Count']

            

nav_res.append('Others')

cont_res.append(cont_sum)





browser_smartphone_result_print = pd.DataFrame({"Browser":nav_res, "Count":cont_res})

    

ax[1, 1].pie(browser_smartphone_result_print.iloc[:]['Count'], labels = browser_smartphone_result_print.iloc[:]['Browser'], autopct='%.0f%%' )

ax[1,1].set_title('Mobile browsers', size = '14')

plt.savefig('OS_and_browser_info.png')


