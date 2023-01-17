import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_cs
import datetime 

## data visualization 
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline





for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("/kaggle/input/new-york-city-bike-share-dataset/NYC-BikeShare-2015-2017-combined.csv")
        
"""

 Objective of the Project: 

1 Group targetting based on most common group- Age Group/Gender/Location/time of year/month/day
 based on cutomer or one time users 
2. Use geospacial data to show where are all bikes located and highlight the one with most common used routes for starting and endind
3. Average and median time by customers/one time users 
4. Figure out which one are more relevant stations (volume) and then look at trends for top five station
5. Most popular trip (https://towardsdatascience.com/citi-bike-2017-analysis-efd298e6c22c: Part 3 )
6. So we can identify which bike would need some attention/replacement - busiest bike. Will help which category of bikes and location need customer care  

7. Predictive Model: Estimate (https://towardsdatascience.com/citi-bike-2017-analysis-efd298e6c22c)



https://github.com/alhankeser/citibike-analysis

"""
# For a good project 

# https://studentwork.prattsi.org/infovis/projects/visualizing-citi-bikes/ 
"""First step is to import and understand what is in the data. 

This step involves looking at the columns and their data type 


"""
df.describe()
df.info()
### fixing the datatypes 
df['ignore'] = ""
df['ignore_reason'] = ""
df[['Start Time','Stop Time']] = df[['Start Time','Stop Time']].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S.%f')
df['Birth Year'] = pd.to_numeric(df['Birth Year'], downcast='integer')
# df.info()



# count = df["Birth Year"].isna().sum() to count number of nas 

## Looking at unique values 

df['Gender'].value_counts()
df['Birth Year'].value_counts(sort= True)
df['User Type'].value_counts(sort= True)
df['Start Station Name'].value_counts(sort= True)  ## Try to put this in a map with count 
df['Trip_Duration_in_min'].value_counts(sort= True)  ## trip duration of 1 minutes makes no sense have to clean up the data 
## multiple 1 values don't make sense
### Looking for duplicated in the data 

duplicates = df.duplicated(keep='first')
df.insert(len(df.columns), "duplicate", duplicates, allow_duplicates = False)
print("Found {} duplicate rows".format(len(df[duplicates])))

### Validating data 

df['Years_old'] = df['Start Time'].dt.year - df['Birth Year'] 

df_subscribers = df[df["User Type"] == "Subscriber"]
df_customers = df[df["User Type"] == "Customer"]
sns.set(style="whitegrid", color_codes=True)
sns.distplot(df_subscribers['Years_old'], kde= False,color = "Green", label = "subscriber", bins = 100 ,rug = True)
sns.distplot(df_customers['Years_old'], kde= False, color = "Red" , label = "customers", bins = 100)
plt.legend(prop={'size': 12})
plt.title('Age of bikers')
plt.xlabel('Age of bikers')
plt.ylabel('Count')

"""
We can clearly see that the population distribution of subcriber is well distributed than the customers. There are also 
some data points where the age is above 80. It is untilike someone above 100 is riding bike, this needs to be 
removed from the data"
"""
df.loc[df['Years_old'] > 80, "ignore_reason"] += "Age invalid"
TIME_RANGES_LIMITS = [0, 5, 15, 30, 45, np.inf]
TIME_RANGES = ["<5", "6-15", "16-30", "31-45", "45+"]

df["Trip_Duration_range"] = pd.cut(df["Trip_Duration_in_min"], TIME_RANGES_LIMITS, labels=TIME_RANGES)

df_subscribers = df[df["User Type"] == "Subscriber"]
df_customers = df[df["User Type"] == "Customer"]
DATAFRAMES = [df, df_subscribers, df_customers]

FONT_SCALE = 1
for i in range(3):
    dfr = DATAFRAMES[i]

    with sns.plotting_context("notebook", font_scale=FONT_SCALE):
        f = sns.countplot(y = "Trip_Duration_range",palette = "Blues", data=DATAFRAMES[i])
        f.get_figure().get_axes()[0]
        f.set(xlabel='Trip Count', ylabel='Trip Duration in Minute')
        plt.show()

"""
 From the results we can infer that for the customers the counts for trip less than 5 minutes is very high, 
this could be ither because glitch first time customers run into (very often) or could be that customers tend 
to take trips less than 5 minutes. This makes little sense. 

For the purpose of analysis, I will get rid of data which has trip time less than 5 minutes 

"""        

df.loc[df['Trip_Duration_in_min'] >= 5, "ignore_reason"] += "Invalid Trip duration"
### Find most common start and end routes 
import folium
df_sub = df[['Start Station Name','End Station Name', 'Start Station Latitude','Start Station Longitude','End Station Latitude','End Station Longitude']]
# df_routes = df_sub.groupby(['Start Station Name','Start Station Latitude', 'Start Station Longitude', 'End Station Name', 'End Station Latitude','End Station Longitude']).size().reset_index(name='Counts of trips')

# df_routes = df_routes.nlargest(50, columns=['Counts of trips'])
# # print(df_routes)

# df_sub = df_sub[df_sub['Start Station Name'] != df_sub['End Station Name']]
df_sub['both'] = df_sub['Start Station Name'] + ', ' + df_sub['End Station Name']
# # perform the transformation asked
df_sub = df_sub.groupby(['Start Station Name','Start Station Latitude', 'Start Station Longitude', 'End Station Name', 'End Station Latitude','End Station Longitude'])['both'].count().reset_index(name='Counts of trips')



df_sub = df_sub.nsmallest(100, columns=['Counts of trips'])
# print(df_sub)

map1 = folium.Map(

    location=[40.732775,-74.105973],
    tiles='cartodbpositron',
    zoom_start= 15,
    max_width= 150, max_height=150
    
#     m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

)


df_sub.apply(lambda row:folium.CircleMarker(location=[row["End Station Latitude"], row["End Station Longitude"]], popup=row["End Station Name"],radius=5, color="Red").add_to(map1), axis=1)
df_sub.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],popup=row["Start Station Name"]).add_to(map1), axis=1)


map1



map1 = folium.Map(

    location=[40.732775,-74.105973],
    tiles='cartodbpositron',
    zoom_start= 15,
    max_width= 250, max_height=250
)
# df_routes.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],radius=2, color="#007849").add_to(map1),axis=1)
df_routes.apply(lambda row:folium.CircleMarker(location=[row["End Station Latitude"], row["End Station Longitude"]], popup=row["End Station Name"],radius=2, color="Red").add_to(map1),axis=1)
df_routes.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],popup=row["Start Station Name"], radius=2, color="#007849").add_to(map1),axis=1)

popup
map1
# # Construct downtown map
# downtown_map = folium.Map(location = 'nashville', zoom_start = 15)
# folium.GeoJson(urban_polygon).add_to(downtown_map)

# # Create popups inside the loop you built to create the markers
# for row in urban_art.iterrows():
#     row_values = row[1] 
#     location = [row_values['lat'], row_values['lng']]
#     popup = (str(row_values['title']) + ': ' + 
#              str(row_values['desc'])).replace("'", "`")
#     marker = folium.Marker(location = location, popup = popup)
#     marker.add_to(downtown_map)

import folium
map1 = folium.Map(
location=[40.785091, -73.968285],
tiles='cartodbpositron',
zoom_start= 10,
width='50%', height='50%'
)
new_start.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],popup=row['Start Station Name'],radius=2, color="#007849").add_to(map1),axis=1)
map1

### Find most common start and end routes 
import folium
df_sub = df[['Start Station Name', 'Start Station Latitude','Start Station Longitude','End Station Name','End Station Latitude','End Station Longitude','Trip_Duration_in_min']]
df_routes = df_sub.groupby(['Start Station Name','Start Station Latitude', 'Start Station Longitude', 'End Station Name', 'End Station Latitude','End Station Longitude'], as_index = False).agg({'Trip_Duration_in_min': ['mean']}).rename(columns={'mean':'Avg_trip_duration','Start Station Name': 'Start Station' , 'End Station Name': 'End Station' })
df_routes.columns = df_routes.columns.droplevel(1)
df_routes.rename(columns = {'Trip_Duration_in_min':'Avg_Trip_Duration'},  inplace=True)
df_routes = df_routes.nlargest(10, columns=['Avg_Trip_Duration'])

map1 = folium.Map(

    location=[40.732775,-74.105973],
    tiles='cartodbpositron',
    zoom_start= 15,
    max_width= 250, max_height=250
)

for row in df_routes.iterrows():
    print(row["End Station Latitude"])
    folium.CircleMarker(location=[row["End Station Latitude"], row["End Station Longitude"]], 
                        popup=row["End Station Name"],radius=2, color="Red")
#     df_routes.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],popup=row["Start Station Name"], radius=2, color="#007849").add_to(map1),axis=1)

df_routes.apply(lambda row:folium.CircleMarker(location=[row["End Station Latitude"], row["End Station Longitude"]], popup=row["End Station Name"],radius=2, color="Red").add_to(map1),axis=1)
df_routes.apply(lambda row:folium.CircleMarker(location=[row["Start Station Latitude"], row["Start Station Longitude"]],popup=row["Start Station Name"], radius=2, color="#007849").add_to(map1),axis=1)


map1
