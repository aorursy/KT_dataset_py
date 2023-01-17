import pandas as pd
import numpy as np
import bq_helper
from bq_helper import BigQueryHelper
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.head("crime", num_rows=3)
bq_assistant.table_schema("crime")
all_crime_query = """ SELECT year, primary_type, COUNT(*) as count
                        FROM `bigquery-public-data.chicago_crime.crime`
                        GROUP BY year, primary_type
                        ORDER BY year, count DESC
                        """

all_crime_data = chicago_crime.query_to_pandas_safe(all_crime_query)

import matplotlib.pyplot as plt 

plt.rcParams['font.size'] = 22 #set default text size
fig, ax = plt.subplots(1, 1, figsize=(25, 15)) 
#loop over the crimes of interest, plotting a line for each one:
for crime in ['THEFT', 'BATTERY', 'ASSAULT', 'CRIMINAL DAMAGE', 'NARCOTICS']:
    ax.plot(all_crime_data.loc[all_crime_data['primary_type'] == crime, ['year']], all_crime_data.loc[all_crime_data['primary_type'] == crime, ['count']], label=crime.lower().capitalize(), linewidth=3.0, alpha = 0.75)

plt.ylabel("Number of reports", size=30, color = 'grey')
plt.xlabel("Year", size=30, color = 'grey')

plt.xticks(np.arange(min(all_crime_data.year)+1, max(all_crime_data.year)+1, step=2), color = 'grey')
plt.yticks(np.arange(0, 110000, 20000), color='grey')
legend = plt.legend(loc=0, fontsize='medium', frameon=False)
plt.setp(legend.get_texts(), color='grey')
plt.show()
monthly_query = """ 
SELECT 
  primary_type,  EXTRACT(MONTH FROM date) AS Month, Count(*) as count
FROM
  `bigquery-public-data.chicago_crime.crime`
WHERE (year >2010) AND (year <2018)
GROUP BY  Month, primary_type
ORDER BY  Month, count DESC
                        """
monthly_data = chicago_crime.query_to_pandas_safe(monthly_query)

fig, ax = plt.subplots(1, 1, figsize=(25, 15))

for crime in ['THEFT', 'BATTERY', 'ASSAULT', 'CRIMINAL DAMAGE', 'NARCOTICS']:
    plt.plot(monthly_data.loc[monthly_data['primary_type'] == crime, ['Month']], monthly_data.loc[monthly_data['primary_type'] == crime, ['count']], label=crime.lower().capitalize(), linewidth=3.0, alpha=0.75)
plt.xticks(np.arange(1, 13, 1), color='grey')
plt.yticks(np.arange(0, 61000, 10000), color='grey')
plt.ylabel("Number of reports", color='grey', size = 30)
plt.xlabel("Month (1=January)", color='grey', size = 30)

legend = plt.legend(loc=1, fontsize='medium', frameon=False)
plt.setp(legend.get_texts(), color='grey')
plt.show()
truncated_2017_query = """ SELECT year, primary_type, COUNT(*) as count
                        FROM `bigquery-public-data.chicago_crime.crime`
                        WHERE (UNIX_SECONDS((SELECT MAX(date)
                                      FROM `bigquery-public-data.chicago_crime.crime`)) - UNIX_SECONDS(date) > UNIX_SECONDS(TIMESTAMP "2018-01-01 00:00:00") - UNIX_SECONDS(TIMESTAMP "2017-01-01 00:00:00"))
                        AND (year = 2017)
                        GROUP BY year, primary_type
                        ORDER BY count DESC
                        """

truncated_2017_data = chicago_crime.query_to_pandas_safe(truncated_2017_query)
truncated_2018_data = all_crime_data.loc[all_crime_data['year'] == 2018, ['primary_type', 'count']]
full_2017_data = all_crime_data.loc[all_crime_data['year'] == 2017, ['primary_type', 'count']]
# extrapolate our 2017 numbers:
theft_forecast = full_2017_data.iloc[0][1]*(truncated_2018_data.iloc[0][1]/ truncated_2017_data.iloc[0][2])
battery_forecast = full_2017_data.iloc[1][1]*truncated_2018_data.iloc[1][1]/ truncated_2017_data.iloc[1][2]
criminal_damage_forecast = full_2017_data.iloc[2][1]*(truncated_2018_data.iloc[2][1] / truncated_2017_data.iloc[2][2])
assault_forecast = full_2017_data.iloc[3][1]*(truncated_2018_data.iloc[3][1] / truncated_2017_data.iloc[3][2])
narcotics_forecast = full_2017_data.iloc[8][1]*(truncated_2018_data.iloc[6][1] / truncated_2017_data.iloc[7][2])

up_to_2017_crime_data = all_crime_data.loc[all_crime_data['year'] != 2018, ['year', 'primary_type', 'count']]
interpolated_crime_data = all_crime_data.loc[all_crime_data['year'] == 2017, ['year', 'primary_type', 'count']]
# add on our forecasted data:
forecast = {'year':[2018, 2018, 2018, 2018, 2018], 'primary_type':["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "NARCOTICS"], 'count':[theft_forecast, battery_forecast, criminal_damage_forecast, assault_forecast, narcotics_forecast]}
forecast_data = pd.DataFrame(data = forecast)
interpolated_crime_data = interpolated_crime_data.append(forecast_data)
fig, ax = plt.subplots(1, 1, figsize=(25, 15))

i=0
for crime in ['THEFT', 'BATTERY', 'ASSAULT', 'CRIMINAL DAMAGE', 'NARCOTICS']:
    ax.plot(up_to_2017_crime_data.loc[up_to_2017_crime_data['primary_type'] == crime, ['year']], up_to_2017_crime_data.loc[up_to_2017_crime_data['primary_type'] == crime, ['count']], label=crime.lower().capitalize(), linewidth=3.0, alpha = 0.75)
    ax.plot(interpolated_crime_data.loc[interpolated_crime_data['primary_type'] == crime, ['year']], interpolated_crime_data.loc[interpolated_crime_data['primary_type'] == crime, ['count']], '--', color='C'+str(i), linewidth=3.0, alpha = 0.75)
    i = i+1 # a counter to ensure our colours match on the interpolated plot
    
plt.xticks(np.arange(min(all_crime_data.year)+1, max(all_crime_data.year)+1, step=2), color='grey')
plt.yticks(np.arange(0, 110000, 20000), color='grey')
plt.ylabel("Number of reports", size = 30, color='grey')
plt.xlabel("Year", size = 30, color='grey')
legend = plt.legend(loc=1, fontsize='medium', frameon=False)
plt.setp(legend.get_texts(), color='grey')
plt.show()
ward_query = """ SELECT year, primary_type, ward, COUNT(*) as count
                        FROM `bigquery-public-data.chicago_crime.crime`
                        WHERE (is_nan(ward) = FALSE) AND ((primary_type = 'ASSAULT') OR (primary_type = 'THEFT')) AND ((year = 2017) OR (year = 2016))
                        GROUP BY year, primary_type, ward
                        ORDER BY year, primary_type, ward
                        """
ward_data = chicago_crime.query_to_pandas_safe(ward_query)
# we could do the following with SQL queries instead, but usually best to do locally:
ward_assault = ward_data.loc[ward_data['primary_type'] == 'ASSAULT', ['year', 'ward', 'count']]
ward_assault_2017 = ward_assault.loc[ward_assault['year'] == 2017, ['ward', 'count']]
ward_assault_2016 = ward_assault.loc[ward_assault['year'] == 2016, ['ward', 'count']]
ward_theft = ward_data.loc[ward_data['primary_type'] == 'THEFT', ['year', 'ward', 'count']]
ward_theft_2017 = ward_theft.loc[ward_theft['year'] == 2017, ['ward', 'count']]
ward_theft_2016 = ward_theft.loc[ward_theft['year'] == 2016, ['ward', 'count']]

fig, ax = plt.subplots(1, 1, figsize=(25, 25))

myx = 100*(np.array(ward_theft_2017['count']) - np.array(ward_theft_2016['count']))/np.array(ward_theft_2016['count'])
myy = 100*(np.array(ward_assault_2017['count']) - np.array(ward_assault_2016['count']))/np.array(ward_assault_2016['count'])
plt.ylabel("Assault change (%)", color='grey', size=30)
plt.xlabel("Theft change (%)", color='grey', size=30)
# plot axes through the origin:
ax.axhline(y=0, color='k', alpha=0.2)
ax.axvline(x=0, color='k', alpha=0.2)

for i in range(50):
    if (abs(myx[i]) > 19) or (abs(myy[i]) > 20): # only plot 'unusual' wards
        ax.annotate("Ward " + str(i), (myx[i], myy[i]), alpha=0.5, xytext = (myx[i] + 0.5, myy[i] + 0.5), size = 18)
ax.scatter(myx, myy, s = 1/50*(np.array(ward_theft_2017['count']) + np.array(ward_assault_2017['count'])))
plt.xticks(np.arange(-40, 50, 10), color='grey')
plt.yticks(np.arange(-40, 50, 10), color='grey')
plt.show()
narc_2017_query = """ SELECT latitude, longitude
                        FROM `bigquery-public-data.chicago_crime.crime`
                        WHERE (primary_type = 'NARCOTICS') AND (year = 2017)
                        """
narc_2017_data = chicago_crime.query_to_pandas_safe(narc_2017_query)
# from bokeh.io import output_file, show
# from bokeh.models import ColumnDataSource, GMapOptions
# from bokeh.plotting import gmap
# output_file("gmap.html")
# map_options = GMapOptions(lat=41.80, lng=-87.65, map_type="roadmap", zoom=11)
# d = gmap("You need to put your own API key here", map_options, title="Chicago Narcotics Violations")
# source = ColumnDataSource(
#     data=dict(mylat=lat,
#               mylon=long)
# )
# d.circle(x="mylon", y="mylat", size=7, fill_color="red", fill_alpha=0.05, line_alpha=0, source=source)
# show(d)
