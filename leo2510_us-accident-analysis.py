# Load necessary library
import os
import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import seaborn as sns
import folium
import branca.colormap as cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
%matplotlib inline

plt.rcParams["figure.figsize"] = (15,8)
# Load and preview data 
accident = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
accident.head()
# Summary Statistics
accident.describe()
# Check each column for nas
accident.isnull().sum()
# Exclude unnecessary columns
exclude = ["TMC","End_Lat","End_Lng","Description","Number","Street","Timezone",
           "Airport_Code","Weather_Timestamp","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight"]
accident_clean = accident.drop(exclude,axis=1)
accident_clean.head()
# Check nas after excluding unnecessary columns
accident_clean.isnull().sum()
# Adding calculation of time difference of start and end time in minutes
accident_clean.Start_Time = pd.to_datetime(accident_clean.Start_Time)
accident_clean.End_Time = pd.to_datetime(accident_clean.End_Time)
accident_clean["Time_Diff"] = (accident_clean.End_Time - accident_clean.Start_Time).astype('timedelta64[m]')

accident_clean["Start_Date"] = accident_clean["Start_Time"].dt.date
accident_clean["End_Date"] = accident_clean["End_Time"].dt.date
accident_clean["Year"] = accident_clean["Start_Time"].dt.year
accident_clean["Month"] = accident_clean["Start_Time"].dt.month
accident_clean["Day"] = accident_clean["Start_Time"].dt.day
accident_clean["Hour"] = accident_clean["Start_Time"].dt.hour

# Excluding accidents in 2015 and 2020 where there's not enough data
accident_clean = accident_clean[(accident_clean["Year"] > 2015) & (accident_clean["Year"] < 2020)]
group = accident_clean.groupby(["Year"]).agg(Count = ('ID','count'))

# Verify data
accident_clean.head()
# Examine data
accident_clean.groupby(["Year","Severity"]).size().unstack()
# accident_clean.groupby(["Start_Date","Severity"])["ID"].count()

# Group by year and Group by year and severity
group_year = accident_clean.groupby(["Year"]).agg(Count = ('ID','count'))
group_year_sev = accident_clean.groupby(["Year","Severity"]).size().unstack()

# YoY Total Accident Count
# fig = plt.figure(figsize=(15,8))

# plt.subplot(1, 2, 1)
plt.plot(group_year.index, group_year["Count"])
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks(np.arange(2016, 2020, 1.0))

plt.show
# YoY trend by severity, more in 2, 1 and 4 looks like flat, need to see in a bar plot
# fig = plt.figure(figsize=(15,8))
group_year_sev2 = accident_clean.groupby(["Year","Severity"]).agg(Count = ('ID','count')).reset_index()
sns.lineplot(x='Year',y='Count',hue="Severity",data=group_year_sev2,palette="Set1")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks(np.arange(2016, 2020, 1.0))
plt.show
# YoY Severity Count
# group_year_sev = accident_clean.groupby(["Year","Severity"]).agg(Count = ('ID','count'))
# group_year_sev
accident_clean.groupby(["Year","Severity"]).size().unstack().plot(kind='bar',stacked=True)
# Makes more sense to show stacked 100%, a different view
accident_clean.groupby(["Year","Severity"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc = 'upper right',title = 'Severity')
plt.show()
# Boxplot to show if temperature has impact on the severity of the accident, 
# looks like the more severe accident has lower temperature
sns.boxplot(x="Severity", y="Temperature(F)", data=accident_clean, palette="Set1")
# Boxplot to show if temperature has impact on the severity of the accident, 
# looks like the more severe accident has lower temperature
sns.boxplot(x="Severity", y="Humidity(%)", data=accident_clean, palette="Set1")
# Examine wind chill and accident severity, lower wind chill cause more severe accidents
sns.boxplot(x="Severity", y="Wind_Chill(F)", data=accident_clean, palette="Set1")
# Count of Severity by Sunrise_Sunset to see if more severe accidents happened at night
pd.crosstab(accident_clean["Severity"], accident_clean["Sunrise_Sunset"], 
            rownames=['Severity'], colnames=['Sunrise_Sunset'])
# Severity 1 and two has same % between day and night while 3 and 4 has more accidents % at nights
accident_clean.groupby(["Severity","Sunrise_Sunset"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc = 'upper right',title = 'Sunrise/Sunset')
plt.show()
# Most accidents happened on the right side of the road
# Severity 3 has more on right then the left side of the road
accident_clean[accident_clean.Side != " "].groupby(["Severity","Side"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)
plt.legend(loc = 'upper right')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()
# Examining Severity by month, most severe accidents (3 and 4) happened in June and July
accident_clean.groupby(["Month","Severity"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True,figsize = (15,8))
plt.legend(loc = 'upper right')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()
sns.distplot(accident_clean['Distance(mi)'])
accident_clean.groupby('Severity')['Distance(mi)'].mean()
# df[(df['year'] > 2012) & (df['reports'] < 30)]
# sns.boxplot(x="Severity", y="Wind_Speed(mph)", 
#             data=accident_clean[accident_clean["Wind_Speed(mph)"] <= 50], palette="Set1")
accident_clean.groupby('Severity')['Time_Diff'].median()
# sns.boxplot(x="Severity", y="Time_Diff", data=accident_clean, palette="Set1")
accident_road = accident[['Severity','Amenity', 'Bump','Crossing','Give_Way',
                         'Junction','No_Exit','Railway','Roundabout','Station',
                         'Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']]
accident_road.head()
accident_road_melt = pd.melt(accident_road,id_vars =['Severity'],value_vars=['Amenity', 'Bump','Crossing','Give_Way',
                         'Junction','No_Exit','Railway','Roundabout','Station',
                         'Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'])
group_road = accident_road_melt.groupby(["Severity","variable","value"]).agg(Count = ('value','count')).reset_index()
group_road.head()
# pd.pivot_table(data=accident_road_melt,index='Severity',columns=['value'],aggfunc='count')
g = sns.catplot(x="Severity", y="Count",
            hue="value", col="variable",
            col_wrap=3, data=group_road, kind="bar",
            height=4, aspect=.7)
g.fig.set_figwidth(15)
g.fig.set_figheight(8)
# Count of True and False of each road condition and group by Severity
(accident_road.set_index('Severity')
 .groupby(level='Severity')
# to do the count of columns nj, wd, wpt against the column ptype using 
# groupby + value_counts
 .apply(lambda g: g.apply(pd.value_counts))
 .unstack(level=1)
 .fillna(0))
# More accidents are happening in the second half of the year
group_day = accident_clean.groupby(["Month","Day"]).size().unstack()
ax = sns.heatmap(group_day, cmap="YlGnBu",linewidths=0.1)
# Most accidents happened between 7 and 8, which is the morning rush hour
# morning rush hour have much more accidents then the afternoon rush hour, which is 4 to 6 in the afternoon

group_hour = accident_clean.groupby(["Day","Hour"]).size().unstack()
ax = sns.heatmap(group_hour, cmap="YlGnBu",linewidths=0.1)
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

# A basic map
# m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)
m = Basemap(llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60)

# m = Basemap(projection='lcc', resolution='h', lat_0=37.5, lon_0=-119,
#             width=1E6, height=1.2E6,
#            llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60)

m.shadedrelief()
m.drawcoastlines(color='gray') 
m.drawcountries(color='gray') 
m.drawstates(color='gray')

lat = accident_clean.Start_Lat.tolist()
lon = accident_clean.Start_Lng.tolist()

x,y = m(lon,lat)
m.plot(x,y,'bo',alpha = 0.2)
# US Shape file from https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip
shapefile_state = '../input/us-shape-data/cb_2018_us_state_500k.shp'


#Read shapefile using Geopandas
gdf_state = gpd.read_file(shapefile_state)
gdf_state.head()
group_state = accident_clean.groupby(["State"]).agg(Count = ('ID','count'))
group_state.reset_index(level=0, inplace=True)
group_state[:5]
# Merge shape file with accident data
state_map = gdf_state.merge(group_state, left_on = 'STUSPS', right_on = 'State')
state_map.head()
# group_state
m = folium.Map(location=[37, -102], zoom_start=4)

folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m)

# myscale = (state_map['Count'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()

m.choropleth(
    geo_data=state_map,
    name='Choropleth',
    data=state_map,
    columns=['State','Count'],
    key_on="feature.properties.State",
    fill_color='YlGnBu',
#     threshold_scale=myscale,
    fill_opacity=1,
    line_opacity=0.2,
    legend_name='Count of Accidents',
    smooth_factor=0
)


style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

toolkit = folium.features.GeoJson(
    state_map,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['State','Count'],
        aliases=['State: ','# of Accidents: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)

m.add_child(toolkit)
m.keep_in_front(toolkit)
folium.LayerControl().add_to(m)

m
variable = ["Severity","Distance(mi)","Time_Diff","Temperature(F)","Wind_Chill(F)","Humidity(%)",
           "Pressure(in)","Visibility(mi)","Wind_Speed(mph)","Precipitation(in)"]
accident_model = accident_clean[variable]
accident_model = accident_model.dropna()
# accident_model['Severity'] = np.where(accident_model['Severity']<=2, 0, 1)
accident_model.head()
Y = accident_model.loc[:,'Severity'].values
X = accident_model.loc[:,'Distance(mi)':'Precipitation(in)'].values

standardized_X = preprocessing.scale(X)
train_x, test_x, train_y, test_y = train_test_split(standardized_X,Y , test_size=0.3, random_state=0)

model = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000)
model.fit(train_x, train_y)
model.score(test_x, test_y)
model_y = model.predict(test_x)

mat = confusion_matrix(test_y,model_y)
sns.heatmap(mat, square=True, annot=True, cbar=False) 
plt.xlabel('predicted value')
plt.ylabel('true value')