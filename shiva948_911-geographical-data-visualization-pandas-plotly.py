import pandas as pd

import numpy as np

import matplotlib as plt

import plotly.plotly as py

import geopandas

from shapely.geometry import Point

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/911.csv")
# total entries , Data types , Memory Usage...

df.info()
# Stastical describtion about 

df.describe()
#prints the head of dataframe- top 5 values by default

df.head()
# Dropping the dummy column 'e' which can done in below 2 ways.

#df.drop('e', axis=1, inplace=True)

del df['e']
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#Splitting the Title

df['reason']=df['title'].apply(lambda i:i.split(':')[0])
df['Detail reason']=df['title'].apply(lambda i:i.split(':')[1])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['Year'] = df['timeStamp'].apply(lambda t: t.year)

df['Date'] = df['timeStamp'].apply(lambda t: t.day)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)
#getting details of station 

df['Station'] = df['desc'].str.extract('(Station.+?);',expand=False).str.strip()
df["day/night"] = df["timeStamp"].apply(lambda x : "night" if int(x.strftime("%H")) > 18 else "day")
df.head()
sns.countplot(x='day/night',data=df)

sns.set_style("darkgrid")
df.Station.value_counts().head(10)
plt.figure(figsize=(20,10))

sns.set_context("paper", font_scale = 2)

sns.countplot(y='Station', data=df, palette="bright", order=df['Station'].value_counts().index[:20])

plt.title("Top 10 Station with highest call")

sns.set_style("darkgrid")

plt.show()

sns.countplot(x='reason',data=df,palette='magma')

sns.set_style("darkgrid")
plt.figure(figsize=(10,8))

sns.countplot(x='Day of Week',data=df,hue='reason',palette='cividis')

plt.title("Calls on each days of the week")

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(x='Month',data=df,hue='reason',palette='hot')

plt.title("Calls Count during each Month ")

sns.set_style("darkgrid")

plt.show()
# Plot for calls recieved monthly combined of all years:

plt.figure(figsize=(12,6))

sns.countplot(x='Month',data=df,palette='spring')

plt.title("Total Calls recieved Monthly For all Years")

sns.set_style("darkgrid")

plt.show()
# Plot for calls recieved yearly:

sns.countplot(x= "Year", data= df,palette='RdYlGn_r')

plt.title("calls recieved on yearly basis")

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(14,7))

sns.set_context("paper", font_scale = 2)

sns.countplot(x= "reason", data= df, palette="bright" ,hue= "Year")

plt.title(" Calls Reason Yearly")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(14,7))

sns.set_context("paper", font_scale = 2)

sns.countplot(x= "Year", data= df, palette="Paired", hue = "reason")

plt.title(" Calls Reason Yearly having the hue of reasons")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("ticks")

plt.show()
plt.figure(figsize=(14,7))

sns.set_context("paper", font_scale = 2)

sns.countplot(x= "Day of Week", data= df, palette="cubehelix", hue= "Year" )     

plt.title(" Daily Calls By Year ")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("white")

plt.show()
plt.figure(figsize=(14,7))

sns.set_context("paper", font_scale = 2)

sns.countplot(x= "Day of Week", data= df, palette="autumn", hue= ("reason") )     

plt.title(" Day Calls By Reason ")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize = (14,7))



sns.set_context("paper", font_scale=2)

sns.countplot(data= df, x= "Month", hue= "Year", palette="gist_earth")



plt.title(" Monthly Calls Yearly")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("ticks")

plt.show()
plt.figure(figsize=(14,7))

sns.set_context("paper", font_scale = 2)

sns.countplot(x= "Month", data= df, palette="copper", hue= "reason")

plt.title(" Monthly Calls Category Combined All Years")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.set_style("darkgrid")

plt.show()
df['Detail reason'].value_counts().head(20)
plt.figure(figsize=(25,10))

sns.set_context("paper", font_scale = 2)

sns.countplot(y='Detail reason', data=df, palette="bright", order=df['Detail reason'].value_counts().index[:20])

plt.title("Top 10 Cases registered")

sns.set_style("darkgrid")

plt.show()
#df.groupby(['zip']).nunique()

df.zip.value_counts().head(5)
plt.figure(figsize=(25,10))

sns.set_context("paper", font_scale = 2)

sns.countplot(x='zip', data=df, palette="seismic_r", order=df['zip'].value_counts().index[:20])

plt.title("top 5 zipcodes for 911 calls")

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(25,10))

sns.set_context("paper", font_scale = 2)

sns.countplot(x='zip', data=df, palette="seismic_r", order=df['zip'].value_counts().index[:20],hue='reason')

plt.title("top 5 zipcodes for 911 calls")

sns.set_style("darkgrid")

plt.show()
df.twp.value_counts().head(10)
plt.figure(figsize=(25,10))

sns.set_context("paper", font_scale = 2)

sns.countplot(y='twp', data=df, palette="spring", order=df['twp'].value_counts().index[:20])

plt.title("top 20 township for 911 calls")

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(25,20))

sns.set_context("paper", font_scale = 2)

sns.countplot(y='twp', data=df, palette="gist_heat", order=df['twp'].value_counts().index[:10],hue='reason')

plt.title("top 20 township for 911 calls with Hue Reason")

sns.set_style("darkgrid")

plt.show()
plt.figure(figsize=(25,100))

sns.set_context("paper", font_scale = 2)

sns.countplot(y='twp', data=df, palette="bright", order=df['twp'].value_counts().index[:5],hue='Detail reason')

plt.title("Top 10 Cases registered")

sns.set_style("darkgrid")

plt.show()
df['Coordinates'] = list(zip(df.lng, df.lat))
df['Coordinates'] = df['Coordinates'].apply(Point)
gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')
gdf.head()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))



# We restrict to South America.

ax = world[world.continent == 'North America'].plot(

    color='white', edgecolor='black')



# We can now plot our GeoDataFrame.

gdf.plot(ax=ax, color='red')

plt.show()
import pandas as pd

from  plotly.offline import plot

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]



data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df['lng'].head(500),

        lat = df['lat'].head(500),

        mode = 'markers',

        marker = dict(

            size = 8,

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'square',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = scl,

            cmin = 0,

            colorbar=dict(

                title="Coordinates points "

            )

        ))]



layout = dict(

        title = '911 calls Location <br>(Hover for co ordinate names)',

        colorbar = True,

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5

        ),

    )



fig = dict( data=data, layout=layout )



iplot( fig, validate=False, filename='d3-airports' )
dayHour = df.groupby(by=['Day of Week','Hour']).count()['reason'].unstack()

dayHour.head()
plt.figure(figsize=(8,4))

sns.heatmap(dayHour,cmap='inferno')

plt.show()
plt.figure(figsize=(8,8))

sns.clustermap(dayHour,cmap='inferno_r')

plt.show()
dayMonth = df.groupby(by=['Day of Week','Month']).count()['reason'].unstack()

dayMonth.head()
sns.heatmap(dayMonth,cmap='Oranges')
sns.clustermap(dayMonth,cmap='Purples')