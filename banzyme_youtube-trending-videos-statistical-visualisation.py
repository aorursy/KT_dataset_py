# Data manipulation libraries
import numpy as np
import pandas as pd
# Data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Initial visualisation styles settings
sns.set(style='darkgrid')
%matplotlib inline
us_videos = pd.read_csv("../input/USvideos.csv")
# Verify that data has been indeed imported
us_videos.head(1)
# Take a quick pick to viee contents of the json file
pd.read_json('../input/US_category_id.json').head(2)['items'][0]
# IMPORT THE JSON CATEGORIES FILE USING THE BUILT-IN json module
import json

# Initialise an empty Categories list as Cat_list
Cat_list = {}
with open("../input/US_category_id.json","r") as jsonData:
    extracted_data = json.load(jsonData)  # Dictionary holding the entire contents of the json file
    
    for x in extracted_data["items"]:
        # id : title 
        Cat_list [ x["id"] ] = x["snippet"]["title"]
Cat_list['1']
us_videos['category_id'] = us_videos['category_id'].apply(lambda y: str(y))
us_videos['Category Title'] = us_videos['category_id'].map(Cat_list)

# Check the last column to verify that we added a new column : 'Category Title
us_videos.head(1)
us_videos['trending_date'] = pd.to_datetime( us_videos['trending_date'], format="%y.%d.%m" )
us_videos['publish_time'] = pd.to_datetime( us_videos['publish_time'] )
type( us_videos['publish_time'][0] )
us_videos.info()
# It might be interesting to add analyse views distribution for different weekdays, months, hours etc.
us_videos['Trending_Year'] = us_videos['trending_date'].apply(lambda x: x.year)
us_videos['Trending_Month'] = us_videos['trending_date'].apply(lambda x: x.month)
us_videos['Trending_Day'] = us_videos['trending_date'].apply(lambda x: x.day)
us_videos['Trending_Day_Of_Week'] = us_videos['trending_date'].apply(lambda x: x.dayofweek)

us_videos["Publish_Year"]=us_videos["publish_time"].apply(lambda y:y.year)
us_videos["Publish_Month"]=us_videos["publish_time"].apply(lambda y:y.month)
us_videos["Publish_Day"]=us_videos["publish_time"].apply(lambda y:y.dayofweek)
us_videos["Publish_Hour"]=us_videos["publish_time"].apply(lambda y:y.hour)
us_videos.head(1)
us_videos['Trending_Day_Of_Week'] [0]
days_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
months_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

us_videos['Trending_Day_Of_Week'] = us_videos['Trending_Day_Of_Week'].map(days_map)
us_videos['Publish_Day'] = us_videos['Publish_Day'].map(days_map)

us_videos['Trending_Month'] = us_videos['Trending_Month'].map(months_map)
us_videos['Publish_Month'] = us_videos['Publish_Month'].map(months_map)
us_videos.head(2)
# 

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
us_videos.iplot(kind='bar', x='Category Title', y='views', title='Views Per Category',mode='markers',size=10)
us_videos.iplot(kind='bar', x='Category Title', y=['likes','dislikes'], title='Number of likes Per Category',mode='markers',size=10, colors=['blue','green'],bargap=0.1)
plt.figure(figsize=(20,8))
us_videos_numerical = us_videos[['views','likes','dislikes' ,'comment_count','Category Title']]
sns.pairplot( us_videos_numerical, hue='Category Title')
sns.heatmap( us_videos_numerical.corr(), cmap='rainbow', annot=True)

plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos, palette='viridis')
plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos[ us_videos['Publish_Day'] == 'Fri' ], hue='Category Title')
plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos[ us_videos['Publish_Day'] == 'Sat' ], hue='Category Title')
# Create a new column to show the dime delta between publish time and trending time
def day_to_trend(x):
    timeDelta = x['trending_date'] - x['publish_time']
    return timeDelta.seconds/3600
us_videos['Days_to_trend'] = us_videos.apply( day_to_trend, axis=1 ) 
sns.distplot(us_videos['Days_to_trend'], bins = 10, color='orange')
us_videos.head(1)
sns.distplot(us_videos['Publish_Hour'], bins = 10, color='purple')
top_10 = us_videos.sort_values('views',ascending=False)[['views','title']]
top_10 = top_10.head(10)
top_10
top_10.iplot(kind='bar', x='title', y='views', title='Top 10 videos',mode='markers')
#sns.barplot(x=top_10.title, y=top_10.views)
