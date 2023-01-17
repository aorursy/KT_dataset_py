from IPython.display import YouTubeVideo
YouTubeVideo('gsoHoem2kl4', width=800, height=450)
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
import os
import bq_helper

# Connect to BigQuery datasets
ny_data_set = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "new_york")
noaa_data_set = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                         dataset_name = "noaa_gsod")

#noaa_data_set.list_tables()
#ny_data_set.table_schema("311_service_requests")
#ny_data_set.head("311_service_requests")
#ny_data_set.head("311_service_requests",selected_columns="location", num_rows=10)
# Define query 
query = """
SELECT 
 Extract(DATE from created_date) AS creation_date, 
 REPLACE(UPPER(complaint_type), "HEATING", "HEAT/HOT WATER") as complaint_type, 
 COUNT(*) AS count 
FROM        `bigquery-public-data.new_york.311_service_requests` 
WHERE
 Extract(YEAR from created_date) = 2016
GROUP BY creation_date, complaint_type 
ORDER BY creation_date ASC, count DESC 
""" 
#ny_data_set.estimate_query_size(query)

# Run query 
complaint_counts = ny_data_set.query_to_pandas_safe(query, max_gb_scanned=0.5)

# Pivot complaint data to create new columns for all of the complaint types 
complaint_counts = complaint_counts.pivot(index='creation_date', columns='complaint_type', values='count')
complaint_counts.columns = [c.lower()
                            .replace(' ', '_')
                            .replace('-', '_') 
                            .replace('/', '_') 
                            for c in complaint_counts.columns]
# Fill zeros for missing values
complaint_counts = complaint_counts.fillna(0)
# Reset index to numeric values for later trending since the date took over the index 
complaint_counts["creation_date"] = complaint_counts.index
complaint_counts.index = range(len(complaint_counts.index))

#print(complaint_counts.head())
# Define query 
query = """
SELECT 
 CAST(CONCAT(w.year,'-',w.mo,'-',w.da) AS date) AS date,
 AVG(w.temp) AS avg_temp,
 MAX(w.max) AS max_temp,
 MIN(w.min) AS min_temp
FROM        `bigquery-public-data.noaa_gsod.gsod2016`  w
INNER JOIN  `bigquery-public-data.noaa_gsod.stations`  s
 ON w.stn=s.usaf
 AND w.wban=s.wban
WHERE
 s.country='US'
 AND s.state = 'NY'
 AND s.name='CENTRAL PARK'
GROUP BY date
ORDER BY date
"""
#noaa_data_set.estimate_query_size(query)

# Run query 
weather_by_day = noaa_data_set.query_to_pandas_safe(query, max_gb_scanned=0.5)

#print(weather_by_day.head(365))
# Create first axis
color = 'tab:orange'
X = complaint_counts.index
y = complaint_counts.urinating_in_public
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_xlabel('time (days)')
ax1.set_ylabel('urine', color=color)
ax1.plot(X, y, color=color)
coefs = poly.polyfit(X, y, 4)
ffit = poly.polyval(X, coefs)
ax1.plot(X, ffit, dashes=[6, 2], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create second axis
color = 'tab:green'
X = weather_by_day.index
y = weather_by_day.avg_temp
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('temp', color=color)
ax2.plot(y, color=color)
coefs = poly.polyfit(X, y, 4)
ffit = poly.polyval(X, coefs)
ax2.plot(X, ffit, dashes=[6, 2], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Display plot
plt.show()
query = """
SELECT 
 REPLACE(UPPER(complaint_type), "HEATING", "HEAT/HOT WATER") as complaint_type, 
 EXTRACT(DAYOFWEEK FROM created_date) AS day_of_week,
 EXTRACT(HOUR FROM created_date)+1 AS hour_of_day,
 latitude,
 longitude
FROM        `bigquery-public-data.new_york.311_service_requests` 
WHERE
     Extract(YEAR from created_date) = 2016
 AND complaint_type = 'Urinating in Public'
 AND latitude IS NOT NULL
"""
#ny_data_set.estimate_query_size(query)

# Run query 
public_urination = ny_data_set.query_to_pandas_safe(query, max_gb_scanned=1)
#public_urination['postal_code'] = pd.to_numeric(public_urination.postal_code, errors='coerce')

#print(public_urination)
plt.figure(figsize=(12, 6))

# Prep map base
map = Basemap(llcrnrlon=-74.1,
              llcrnrlat=40.6,
              urcrnrlon=-73.7,
              urcrnrlat=40.90,
              resolution = 'f')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='white',lake_color='aqua', zorder=1) # zorder keeps this behind scatter points
map.drawcoastlines()

# Plot point on map
x, y = map(public_urination['longitude'], public_urination['latitude'])
map.scatter(x, y, s=1, color='#FF8C00', zorder=2)

#Display map
plt.title('Public Urination occurrences', fontsize=20)
plt.show()
ax = sns.jointplot(x='day_of_week', 
                   y='hour_of_day', 
                   data=public_urination, 
                   kind="kde", 
                   ratio=4, size=8, space=0)

#ax = sns.jointplot(x='hour_of_day', 
#                   y='day_of_week', 
#                   data=public_urination,
#                   kind='hex', 
#                   gridsize=20,
#                   space=1)

#ax = public_urination.plot.hexbin(x='day_of_week', 
#                                  y='hour_of_day', 
#                                  gridsize=15, 
#                                  figsize=(12, 6),
#                                  fontsize=16)

# remove the boundaries around the outside of the plot
sns.despine(bottom=True, left=True)
# Calculate heating and cooling degree days 
change_point = 65
weather_by_day['raw_degree_day_calc'] = change_point - weather_by_day['avg_temp']
weather_by_day['HDD'] = abs(weather_by_day.loc[weather_by_day.raw_degree_day_calc>0,'raw_degree_day_calc'])
weather_by_day['HDD'] = weather_by_day['HDD'].fillna(0)
weather_by_day['CDD'] = abs(weather_by_day.loc[weather_by_day.raw_degree_day_calc<0,'raw_degree_day_calc'])
weather_by_day['CDD'] = weather_by_day['CDD'].fillna(0)
#weather_by_day['total_degree_days'] = weather_by_day['HDD'] + weather_by_day['CDD']

#print(weather_by_day.head(200))
plt.figure(figsize=(12, 5))
plt.plot(weather_by_day.HDD, color='tab:red')
plt.plot(weather_by_day.CDD, color='tab:blue')
plt.plot(weather_by_day.avg_temp, color='tab:green')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.show()
# Define columns for new data frame that holds correlation coefficients for complaint types 
corr_types = ['HDD','CDD','avg_temp']
complaint_corr = pd.DataFrame(columns=['complaint_type']+corr_types)
# add in complaint types and temp types 
complaint_corr['complaint_type'] = complaint_counts.drop(['creation_date'],axis=1).columns
complaint_corr = pd.melt(complaint_corr, id_vars=['complaint_type'], 
                         var_name = 'temp_type', value_name = 'corrcoef')
# calculate correlation coefficients against many weather data types for each complaint type 
for index, row in complaint_corr.iterrows():
    row['corrcoef'] = np.corrcoef(complaint_counts[row['complaint_type']],
                                  weather_by_day[row['temp_type']])[0, 1]

# Find the top complaint types that correlate the most
complaint_corr = complaint_corr.sort_values(by='corrcoef', ascending=False)
complaint_corr = complaint_corr.drop_duplicates(subset='complaint_type', keep="first")
complaint_corr.reset_index(drop=True, inplace=True)
top_complaint_corr = complaint_corr.head(30)

print(top_complaint_corr)
# Specify the chart types I want to create for the various temp types 
chart_types = pd.DataFrame([('Occurs mostly in cold temperature (degree days)','HDD','tab:red'),
                            ('Occurs mostly in warm temperature (degree days)','CDD','tab:blue'),
                            ('Correlation to be higher in warm temperature','avg_temp','tab:green')],
                          columns=['title','temp_type','color'])

# Iterate through the chart type to create them
for index, row in chart_types.iterrows():
    custom_legend_lines = list()
    
    # Plot complaint types that had high correlation to the temp type
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('count of complaints')
    for index2, row2 in top_complaint_corr.iterrows():
        if row['temp_type'] == row2['temp_type']:
            color = plt.cm.gist_ncar(np.random.random())
            X = complaint_counts.index
            y = complaint_counts[row2['complaint_type']]
            ax1.plot(X, y, label=row2['complaint_type'], color=color)
            custom_legend_lines.append(Line2D([0], [0], color=color, 
                                              lw=1, label=row2['complaint_type']))
    
    # Plot temp type
    color=row['color']
    X = weather_by_day.index
    y = weather_by_day[row['temp_type']]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(row['temp_type'], color=row['color'])
    ax2.plot(y, label=row['temp_type'], color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=row['color'])
    custom_legend_lines.append(Line2D([0], [0], color=color, 
                                      lw=3, label=row['temp_type']))
    
    # Finalize chart 
    plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=1, mode="expand", borderaxespad=0.)
    plt.title(row['title'])
    plt.show()
# Specify the chart types I want to create for the various temp types 
chart_types = pd.DataFrame([('Occurs mostly in cold temperature (degree days)','HDD','tab:red'),
                            ('Occurs mostly in warm temperature (degree days)','CDD','tab:blue'),
                            ('Correlation to be higher in warm temperature','avg_temp','tab:green')],
                          columns=['title','temp_type','color'])

# Iterate through the chart type to create them
for index, row in chart_types.iterrows():
    custom_legend_lines = list()
    
    # Plot complaint types that had high correlation to the temp type
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('complaint range - 0 to 1')
    for index2, row2 in top_complaint_corr.iterrows():
        if row['temp_type'] == row2['temp_type']:
            color = plt.cm.gist_ncar(np.random.random())
            X = complaint_counts.index
            # this is the main line that changed from the prior code to normalize the chart results 
            y = complaint_counts[row2['complaint_type']]/complaint_counts[row2['complaint_type']].max()
            ax1.plot(X, y, label=row2['complaint_type'], color=color)
            custom_legend_lines.append(Line2D([0], [0], color=color, 
                                              lw=1, label=row2['complaint_type']))
    
    # Plot temp type
    color=row['color']
    X = weather_by_day.index
    y = weather_by_day[row['temp_type']]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(row['temp_type'], color=row['color'])
    ax2.plot(y, label=row['temp_type'], color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=row['color'])
    custom_legend_lines.append(Line2D([0], [0], color=color, 
                                      lw=3, label=row['temp_type']))
    
    # Finalize chart 
    plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=1, mode="expand", borderaxespad=0.)
    plt.title(row['title'])
    plt.show()