#import libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



crime= pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='unicode_escape')

crime.info()
crime.describe()
crime.head()
crime.SHOOTING.unique()  # If there is a shot, it is filled with 'Y'. For this reason, I filled all the missing values with 'N'.
crime = crime.drop(['INCIDENT_NUMBER','OFFENSE_CODE'], axis=1)



#replace NaN values with 'N' : means No

crime.SHOOTING.fillna('N', inplace=True)



#Replace -1 values in Lat/Long with NaN

crime.Lat.replace(-1, None, inplace=True)

crime.Long.replace(-1, None, inplace=True)





#change the column names

rename = {'OFFENSE_CODE_GROUP':'Offense_group',

         'OFFENSE_DESCRIPTION':'Description',

         'DISTRICT':'District',

         'REPORTING_AREA':'Area',

         'SHOOTING':'Shooting',

         'OCCURRED_ON_DATE':'Date',

         'YEAR':'Year',

         'MONTH':'Month',

         'DAY_OF_WEEK':'Day',

         'HOUR':'Hour',

         'UCR_PART':'UCR',

         'STREET':'Street'}

crime.rename(index=str, columns=rename, inplace=True)



#setting the index to be the date will help 

crime.index = pd.DatetimeIndex(crime.Date)
crime_dist = crime.copy()
crime.District.unique()
district_name = {

'D14':'Brighton',

'C11':'Dorchester',

'D4':'South End',

'B3':'Mattapan',

'B2':'Roxbury',

'C6':'South Boston',

'A1':'Downtown',

'E5':'West Roxbury',

'A7':'East Boston',

'E13':'Jamaica Plain',

'E18':'Hyde Park',

'A15':'Charlestown'

}

crime['District'] = crime['District'].map(district_name)
crime.head()
crime.resample('M').size()
monthly = crime.resample('M').size().to_frame(name='Count')

monthly = monthly.reset_index(level='Date')

import plotly.express as px



fig = px.line(monthly, x=monthly.Date, y=monthly.Count)

fig.update_layout(title='Number of Crimes per Month(2015-2018)',

                   xaxis_title='Month',

                   yaxis_title='Number of Crimes')

fig.show()



results = px.get_trendline_results(fig)

print(results)
#Categorization 

#violation.Offense_group.unique()

#violation.Offense_group.value_counts()



import re

violation = crime[crime.Offense_group.str.contains('Violation')]

assault = crime[crime.Offense_group.str.contains('Assault')]

burglary = crime[crime.Offense_group.str.contains('Burglary')]

harassment = crime[crime.Offense_group.str.contains('Harassment')]

larceny = crime[crime.Offense_group.str.contains('Larceny|Theft')]

larceny = larceny[~larceny.Offense_group.str.contains('Recovery')]

#Investigate = crime[crime.Offense_group.str.contains('Investigate|Search', flags=re.IGNORECASE, regex=True)]

killing = crime[crime.Offense_group.str.contains('Manslaughter|Homicide')] #bunlar az sayıdaydı ancak en ciddi suçlardandı az row olması sebebiyle de aynı categoriye aldım

fraud = crime[crime.Offense_group.str.contains('Confidence Games|Fraud|Counterfeiting')]

#x = crime[crime.Offense_group.str.contains('Missing Person Located', flags=re.IGNORECASE, regex=True)] it was not crime

mv_accident = crime[crime.Offense_group.str.contains('Accident')]

#medicaid = crime[crime.Offense_group.str.contains('Medical Assistance')]

robbery = crime[crime.Offense_group.str.contains('Robbery')]

disputes = crime[crime.Offense_group.str.contains('Verbal Disputes')]

vandalism = crime[crime.Offense_group.str.contains('Vandalism')]



#A column named Category has been added for each crime category.

violation.insert(0, 'Category', 'Violation')

assault.insert(0, 'Category', 'Assault')

burglary.insert(0, 'Category', 'Burglary')

harassment.insert(0, 'Category', 'Harrassment')

larceny.insert(0, 'Category', 'Larceny')

#Investigate = crime[crime.Offense_group.str.contains('Investigate|Search', flags=re.IGNORECASE, regex=True)]

killing.insert(0, 'Category', 'Killing') #bunlar az dayıdaydı ancak en ciddi suçlardandı az row olması sebebiyle de aynı categoriye aldım

fraud.insert(0, 'Category', 'Fraud')

#x = crime[crime.Offense_group.str.contains('Missing Person Located', flags=re.IGNORECASE, regex=True)] it was not crime

mv_accident.insert(0, 'Category', 'Motor vehicle accident')

#medicaid = crime[crime.Offense_group.str.contains('Medical Assistance')]

robbery.insert(0, 'Category', 'Robbery')

disputes.insert(0, 'Category', 'Verbal disputes')

vandalism.insert(0, 'Category', 'Vandalism')



#A dataframe called categorized_crimes indicativing categorized crimes was created.

frames = [violation, assault, burglary, harassment, larceny, killing, fraud, mv_accident, robbery, disputes, vandalism]

categorized_crimes = pd.concat(frames)

categorized_crimes
categorized_crimes['Date'] = pd.to_datetime(categorized_crimes['Date'])

monthly_crimes_count = categorized_crimes.pivot_table(index=pd.Grouper(freq = 'M', key ='Date'), columns='Category', aggfunc=np.size, values= 'Offense_group')

monthly_crimes_count

monthly_crimes_count = monthly_crimes_count.reset_index(level='Date')

monthly_crimes_count = pd.melt(monthly_crimes_count, id_vars=['Date'])

monthly_crimes_count.tail()


fig = px.scatter(monthly_crimes_count, x='Date', y=monthly_crimes_count.value, color = 'Category', trendline="ols")



fig.update_layout(title='Number of Crimes for each Crime Category',

                   yaxis_title='Number of Crimes')

fig.show()



results = px.get_trendline_results(fig)

print(results)

plt.figure(figsize=(10,8))

x = crime.Offense_group.value_counts().head(30).plot(kind='bar', color = 'salmon')
crime.District.value_counts()
crime['Date'] = pd.to_datetime(crime['Date'])

district_crimes_count = crime.pivot_table(index=pd.Grouper(freq = 'M', key ='Date'), columns='District', aggfunc=np.size, values= 'Offense_group')



district_crimes_count = district_crimes_count.reset_index(level='Date')

district_crimes_count = pd.melt(district_crimes_count, id_vars=['Date'])

district_crimes_count


fig = px.scatter(district_crimes_count, x='Date', y=district_crimes_count.value, color = 'District', trendline="ols")



fig.update_layout(title='Number of Crimes for each District',

                   yaxis_title='Number of Crimes')

fig.show()



results = px.get_trendline_results(fig)

print(results)
plt.figure(figsize=(20,10))

sns.countplot(x = 'Category', data = categorized_crimes, order = categorized_crimes.Category.value_counts().index, hue='District')
plt.figure(figsize=(10,6))

sns.countplot(x='Month',data=crime,palette="Reds")
crime_2015=crime[crime['Year']==2015]

crime_2016=crime[crime['Year']==2016]

crime_2017=crime[crime['Year']==2017]

crime_2018=crime[crime['Year']==2018]



fig, axes = plt.subplots(1,4, figsize = (32,6))



sns.countplot(x='Month',data=crime_2015,palette = 'Blues',ax = axes[0]).set_title('2015')

sns.countplot(x='Month',data=crime_2016,palette = 'Greens',ax = axes[1]).set_title('2016')

sns.countplot(x='Month',data=crime_2017,palette = 'Reds',ax = axes[2]).set_title('2017')

sns.countplot(x='Month',data=crime_2018,palette = 'Blues',ax = axes[3]).set_title('2018')

print(crime_2016.Offense_group.count())

print(crime_2017.Offense_group.count())
all_catcrime = ['Larceny','Motor vehicle accident', 'Violation', 'Assault', 'Vandalism', 'Verbal disputes', 'Fraud','Killing', 'Burglary', 'Robbery','Harrassment']

catcrime_month = categorized_crimes.copy()

catcrime_month = catcrime_month[catcrime_month['Category'].isin(all_catcrime)]

catcrime_month= catcrime_month.groupby(['Month','Category']).size().reset_index(name = 'Number of Crimes')

catcrime_month['Months'] = catcrime_month['Month']



#pivot table

catcrime_month_reshape = pd.pivot_table(catcrime_month, index=['Month'], columns=['Category'], values='Number of Crimes', aggfunc=np.sum)





catcrime_month_reshape.plot(kind= 'bar', stacked = True, figsize=(15,8),color=['#c8553d','#370617','#6a040f','#9d0208','#f9844a','#450920','#dc2f02','#e85d04','#f48c06','#90be6d','#4d908e'])

plt.title('Number of Crimes by Type')

plt.show()
plt.figure(figsize=(10,8))

categorized_crimes.Category.value_counts().head(10).plot(kind='bar', color = 'salmon')
catcrime_time = categorized_crimes.copy()

top10_catcrime = ['Larceny','Motor vehicle accident', 'Violation', 'Assault', 'Vandalism', 'Verbal disbutes', 'Fraud', 'Burglary', 'Robbery','Harrassment']

catcrime_time = catcrime_time[catcrime_time['Category'].isin(all_catcrime)]

catcrime_time = catcrime_time.groupby('Hour').size().reset_index(name = 'Number of Crimes')

catcrime_time['Hour'] = catcrime_time['Hour'].apply(lambda i: str(i)+':00')

catcrime_time
plt.figure(figsize=(15,8))

sns.pointplot(data = catcrime_time, x = 'Hour', y = 'Number of Crimes')

plt.show()

#group crimes 

catcrime_type = categorized_crimes.copy()

catcrime_type = catcrime_type[catcrime_type['Category'].isin(top10_catcrime)]

catcrime_type= catcrime_type.groupby(['Hour','Category']).size().reset_index(name = 'Number of Crimes')

catcrime_type['Hours'] = catcrime_type['Hour'].apply(lambda i: str(i)+':00')



#pivot table

catcrime_type_reshape = pd.pivot_table(catcrime_type, index=['Hour'], columns=['Category'], values='Number of Crimes', aggfunc=np.sum)





catcrime_type_reshape.plot(kind= 'bar', stacked = True, figsize=(15,8),color=['#450920','#370617','#6a040f','#9d0208','#f9844a','#dc2f02','#e85d04','#f48c06','#90be6d','#4d908e'])

plt.title('Number of Crimes by Type')

plt.show()
crime_days = crime.groupby('Day').agg('count')

day_counts = crime_days.Offense_group

day_counts
sns.lmplot('Lat', 

           'Long',

           data=crime[:],

           fit_reg=False, 

           hue = 'District',

           palette ='Dark2',

           height=12,

           ci=2,

           scatter_kws={"marker": "D", 

                        "s": 10})

ax = plt.gca()

ax.set_title("All Crime Distribution per District")
sns.lmplot(x="Lat",

           y="Long",

           col="Category",

           hue = 'District',

           data=categorized_crimes.dropna(), 

           col_wrap=2, height=6, fit_reg=False, 

           sharey=False,

           scatter_kws={"marker": "D",

                            "s": 10})
crime_dist
from urllib.request import urlopen

import json

with open('../input/police-district/Police_Districts.geojson') as f:

    boston_geojson1 = json.load(f)



boston_geojson1['features'][0]
dist = pd.DataFrame(data= crime_dist.District.value_counts().values, index= crime_dist.District.value_counts().index, columns=['Count'])

dist = dist.reset_index()

dist.rename({'index': 'District'}, axis='columns', inplace=True)

dist
import folium

from folium import Choropleth, Circle, Marker, plugins

from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster, HeatMapWithTime



crime_map = folium.Map(location=[42.361145,-71.057083], tiles='cartodbpositron', zoom_start=11.2)



crime_map.choropleth(

    geo_data= boston_geojson1,

    data= dist,

    columns= ['District', 'Count'],

    key_on='feature.properties.DISTRICT',

    fill_color='GnBu', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Choropleth of Crimes per Police District'

)

crime_map
crimes = crime.copy()

crimes.dropna( axis = 0, subset = [ 'Lat', 'Long' ], inplace = True )



lats = list(crimes.Lat)

longs = list(crimes.Long)

locations = [lats,longs]



m = folium.Map(location=[42.361145,-71.057083], tiles='cartodbpositron', zoom_start=11.2)



FastMarkerCluster(data=list(zip(lats, longs))).add_to(m)



m.choropleth(

    geo_data= boston_geojson1,

    name='choropleth',

    data= dist,

    columns= ['District', 'Count'],

    key_on='feature.properties.DISTRICT',

    fill_color='YlOrRd', 

    fill_opacity=0.4, 

    line_opacity=0.2,

    legend_name='Distribution of Crimes over the City',

    highlight=False

    )

m
categorized_crime = categorized_crimes.copy()

districts_name = {

'Brighton':'D14',

'Dorchester':'C11',

'South End':'D4',

'Mattapan':'B3',

'Roxbury':'B2',

'South Boston':'C6',

'Downtown':'A1',

'West Roxbury':'E5',

'East Boston':'A7',

'Jamaica Plain':'E13',

'Hyde Park':'E18',

'Charlestown':'A15'

}

categorized_crime['District'] = categorized_crime['District'].map(districts_name)

categorized_crime
crimes_overcity = categorized_crime.pivot_table(index=pd.Grouper(key ='District'), columns='Category', aggfunc=np.size, values= 'Offense_group')

crimes_overcity = crimes_overcity.reindex(['A15', 'A7', 'A1', 'C6','D4', 'D14', 'E13', 'E5','B3', 'C11', 'E18', 'B2'])

crimes_overcity = crimes_overcity.reset_index(level='District')

crimes_overcity = pd.melt(crimes_overcity, id_vars=['District'])

crimes_overcity
import plotly.express as px



fig = px.choropleth_mapbox(crimes_overcity, geojson=boston_geojson1,  featureidkey ='properties.DISTRICT', locations='District', color='value',

                           color_continuous_scale=[[0.0, "rgb(165,0,38)"],

                [0.1111111111111111, "rgb(215,48,39)"],

                [0.2222222222222222, "rgb(244,109,67)"],

                [0.3333333333333333, "rgb(253,174,97)"],

                [0.4444444444444444, "rgb(254,224,144)"],

                [0.5555555555555556, "rgb(224,243,248)"],

                [0.6666666666666666, "rgb(171,217,233)"],

                [0.7777777777777778, "rgb(116,173,209)"],

                [0.8888888888888888, "rgb(69,117,180)"],

                [1.0, "rgb(49,54,149)"]],

                           animation_frame=crimes_overcity["Category"],

                           range_color=(900, 10000),

                           mapbox_style="carto-positron",

                           zoom=10, center={"lat": 42.361145, "lon": -71.057083},

                           opacity=0.5,

                           labels={'value':'Number of Crimes'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()