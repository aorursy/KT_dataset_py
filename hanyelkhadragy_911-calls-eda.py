import datetime

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import cufflinks as cf



import folium

from branca.element import Figure

from geopy.geocoders import Nominatim



%matplotlib inline



# Must enable in order to use plotly off-line (vs. in the cloud)

plotly.offline.init_notebook_mode()

cf.go_offline()
def add_value_labels(ax, spacing=5, top=False, right=False, left=False, annotation=True):

    """Add labels to the end of each bar in a bar chart.



    Arguments:

        ax (matplotlib.axes.Axes): The matplotlib object containing the axes

            of the plot to annotate.

        spacing (int): The distance between the labels and the bars.

    """     

    if not top:

        ax.spines["top"].set_visible(False)

    if not right:    

        ax.spines["right"].set_visible(False)

    if not left:    

        ax.spines["left"].set_visible(False)   

    

    if annotation:

        # For each bar: Place a label

        for rect in ax.patches:

            # Get X and Y placement of label from rect.

            y_value = rect.get_height()

            x_value = rect.get_x() + rect.get_width() / 2



            # Number of points between bar and label. Change to your liking.

            space = spacing

            # Vertical alignment for positive values

            va = 'bottom'



            # If value of bar is negative: Place label below bar

            if y_value < 0:

                # Invert space to place label below

                space *= -1

                # Vertically align label at top

                va = 'top'



            # Use Y value as label and format number with one decimal place

            label = "{:.2f}".format(y_value)



            # Create annotation

            ax.annotate(

                label,                      # Use `label` as label

                (x_value, y_value),         # Place label at end of the bar

                xytext=(0, space),          # Vertically shift label by `space`

                textcoords="offset points", # Interpret `xytext` as offset in points

                ha='center',                # Horizontally center label

                va=va)                      # Vertically align label differently for

                                            # positive and negative values.
df = pd.read_csv('/kaggle/input/montcoalert/911.csv')
df.head()
df.shape
df.columns
df.drop(columns='e',axis=1,inplace=True)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day Of The Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['Date'] = df['timeStamp'].apply(lambda time : time.date())

df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])



dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day Of The Week'] = df['Day Of The Week'].map(dmap)
df.info()
df.head()
# Top 10 Zipcodes

top_zipcode = pd.DataFrame(df['zip'].value_counts().head(10)).reset_index()

top_zipcode.rename(columns={'index':'Zip', 'zip':'Count'},inplace=True)

top_zipcode['Zip'] = top_zipcode['Zip'].apply(lambda zip: int(zip))

top_zipcode
plt.figure(figsize=(10,6))

ax = sns.barplot(x='Zip', y='Count', data=top_zipcode)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

ax.set_xlabel("Zip Code",fontsize=15)

ax.set_ylabel("Count",fontsize=15)

ax.set_title('Zip Code Call Count',fontsize=20)

# Call the function above. All the magic happens there.

add_value_labels(ax,left=True)

plt.xticks(fontsize=11)

plt.show()
# Top 10 Townships 

top_townships = pd.DataFrame(df['twp'].value_counts().head(10)).reset_index()

top_townships.rename(columns={'index':'Township', 'twp':'Count'},inplace=True)

top_townships
plt.figure(figsize=(10,6))

ax = sns.barplot(x='Count', y='Township', data=top_townships)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

ax.set_xlabel("Count",fontsize=15)

ax.set_ylabel("Township",fontsize=15)

ax.set_title('Township Call Count',fontsize=20)



# Call the function above. All the magic happens there.

add_value_labels(ax,left=True,annotation=False)

#plt.xticks(fontsize=11)

plt.show()
df['Reason'].value_counts()
plt.figure(figsize=(10,6))

ax = sns.countplot(x=df['Reason'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

ax.set_xlabel("Reason",fontsize=18)

ax.set_ylabel("Count",fontsize=18)

ax.set_title('Reason Call Count',fontsize=20)

# Call the function above. All the magic happens there.

add_value_labels(ax,left=True)

plt.show()
plt.figure(figsize=(14,8))

ax = sns.countplot(x='Day Of The Week',hue='Reason',data=df)



# Call the function above. All the magic happens there.

add_value_labels(ax,annotation=False,left=True)

plt.xticks(fontsize=14,rotation=60)

plt.yticks(fontsize=14)

ax.set_xlabel("Day Of The Week",fontsize=18)

ax.set_ylabel("Count",fontsize=18)

ax.set_title('Day Of The Week Call Count',fontsize=20)

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 15})

plt.show()
plt.figure(figsize=(14,8))

ax = sns.countplot(x='Month',hue='Reason',data=df)



# Call the function above. All the magic happens there.

add_value_labels(ax,annotation=False,left=True)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

ax.set_xlabel("Month",fontsize=18)

ax.set_ylabel("Count",fontsize=18)

ax.set_title('Monthly Call Count',fontsize=20)



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 15})



plt.show()
byMonth = df.groupby(by='Month').count()

byMonth
layout1 = cf.Layout(

    height=500,

    width=1000

)

byMonth['lat'].iplot(title='Monthly Call Trend')
layout1 = cf.Layout(

    height=500,

    width=1200

)



df.groupby(by='Date').count()['lat'].iplot(title='Daily Call Trend',colors='blue')
layout1 = cf.Layout(

    height=500,

    width=1200

)



df.loc[ (df['Reason'] == 'EMS') ].groupby(by='Date').count()['Reason'].iplot(title='EMS Reason Call Count',colors='blue')
layout1 = cf.Layout(

    height=500,

    width=1200

)



df.loc[ (df['Reason'] == 'Traffic') ].groupby(by='Date').count()['Reason'].iplot(title='Traffic Reason Call Count',colors='blue')
layout1 = cf.Layout(

    height=500,

    width=1200

)



df.loc[ (df['Reason'] == 'Fire') ].groupby(by='Date').count()['Reason'].iplot(title='Fire Reason Call Count',colors='blue')
x = (df.loc[ (df['Reason'] == 'Fire') ].groupby(by='Date').count()['lat'] )

x[lambda x: x >= 250]
import datetime



Top_Fire_Dates = df.loc[ (df['Date'].isin(x[lambda x: x >= 250].index)) & (df['Reason'] == 'Fire'), ['lat','lng','Date']]

Top_Fire_Dates['Count'] = Top_Fire_Dates['Date'].map(x[lambda x: x >= 250])
Top_Fire_Dates.loc[Top_Fire_Dates['Date'] == datetime.date(2020, 6, 3)]
address = 'Philadelphia'

geolocator = Nominatim(user_agent="911_EDA",timeout=30)

location = geolocator.geocode(address,timeout=30)

latitude = location.latitude

longitude = location.longitude

#print('The geograpical coordinate of Philadelphia are {}, {}.'.format(latitude, longitude))



fig=Figure(width=1000,height=500)



# create map of Philadelphia using latitude and longitude values

map_us = folium.Map(location=[latitude, longitude], zoom_start=9, min_zoom=5, max_zoom=14)

fig.add_child(map_us)





date1= datetime.date(2020, 6, 3)



# add markers to map

for lat, lng, Count in zip(Top_Fire_Dates.loc[Top_Fire_Dates['Date'] == date1,'lat'], Top_Fire_Dates.loc[Top_Fire_Dates['Date'] == date1,'lng'], Top_Fire_Dates.loc[Top_Fire_Dates['Date'] == date1,'Count']):

    folium.CircleMarker(

        [lat, lng],

        radius=6,

        tooltip=str(Count)+' Fire Calls',

        color='blue',

        fill_color='#3186cc',

        fill_opacity=0.7).add_to(map_us)  

fig
dayHour = df.groupby(by=['Day Of The Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(16,9))

sns.heatmap(dayHour,cmap='viridis')
plt.figure(figsize=(16,9))

sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day Of The Week','Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(16,9))

sns.heatmap(dayMonth,cmap='viridis')
plt.figure(figsize=(16,9))

sns.clustermap(dayMonth,cmap='viridis')
address = 'Philadelphia'

geolocator = Nominatim(user_agent="911_EDA",timeout=30)

location = geolocator.geocode(address,timeout=30)

latitude = location.latitude

longitude = location.longitude

#print('The geograpical coordinate of Philadelphia are {}, {}.'.format(latitude, longitude))



fig=Figure(width=1000,height=500)



# create map of Philadelphia using latitude and longitude values

map_us = folium.Map(location=[latitude, longitude], zoom_start=9, min_zoom=5, max_zoom=11)

fig.add_child(map_us)



x = df.loc[ (df['lat'].isin(df['lat'].value_counts().head(20).index.to_list())), ['lat','lng'] ]

top_20_lat = pd.DataFrame(x).reset_index()

top_20_lat.drop('index', axis=1, inplace=True)

top_20_lat['Count'] = top_20_lat['lat'].map(top_20_lat['lat'].value_counts())





# add markers to map

for lat, lng, Count in zip(top_20_lat['lat'], top_20_lat['lng'], top_20_lat['Count']):

    folium.CircleMarker(

        [lat, lng],

        radius=6,

        tooltip=str(Count)+' Calls',

        color='blue',

        fill_color='#3186cc',

        fill_opacity=0.7).add_to(map_us)  

fig    