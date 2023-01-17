# This notebook analyses the NYPD Motor Vehicle Collisions open data on Kaggle. 

# The notebook was scheduled to run on PythonEverywhere Cloud service 

# everyday for free until Jan 2019 and update the results. 



# Data processing and analysis:



# First, the data is cleaned and prepared for analysis. 



# Next, a visualization of the number of persons injured in NY, over all 

# years, is made using Plotly interactive visualization tool. Additionally,

# a scattered visualization is created using basemap and matplotlib for fun. 



# In the next step, the focus is shifted to the Manhattan area, the financial 

# hub of NY. A hexbin density plot is made to show which region within Manhattan 

# has suffered from maximum collisions, all years combined. 



# Then, the collision trends in this area are studied as a function of time. The 

# data is divided into 5 main regions within Manhattan. A comparison of 

# the trends in the number of collisions from 2012 to present between the 5 

# regions is made. This gives insight into various aspects such as whether road

# safety in these regions is improving with time, which region has collision 

# safe (relatively speaking) while which one needs more work to improve road safety, etc. 

# A similar comparison between the 5 regions is made to show the trends in the 

# number of persons injured per 100 collisions. This shows which regions suffer 

# from more dangerous collisions that lead to injuries. 
!pip install pyjanitor

#To clean up column names
!pip install basemap

#To plot data on geographical map
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Define function to load the data. Read zip code properly as string.



def load_NYPD_collisions_data():

    csv_path="../input/nypd-motor-vehicle-collisions.csv"

    return pd.read_csv(csv_path, dtype={'ZIP CODE': 'str'})



#Call the function to read the data

Collisions_data=load_NYPD_collisions_data()

#Collisions_data.describe()

Collisions_data.info()

######Clean data names, columns, rows

# Clean up column names

import janitor

Col_data_clean=Collisions_data.clean_names()



#Drop rows with nan

Col_data_clean2=Col_data_clean.dropna(subset=["number_of_persons_injured", "number_of_persons_killed","zip_code"])

#Col_data_clean2.info()



#Select data within NYC by constraining latitude and longitude

df=Col_data_clean2[Col_data_clean2["number_of_persons_injured"]>-1]

df=df[(df["latitude"]>40.49) & (df["latitude"]<40.91)]

df=df[(df["longitude"]>-74.25) & (df["longitude"]<-73.7)]

print("Cleaned up data, after excluding nan and 0 persons injured:")

#df.describe()

df.head(3)
# Geographically visualize data, without using basemap



import matplotlib

import matplotlib.pyplot as plt

import numpy as np



#There are too many rows where number of person injured is 0. 

#They overwhelm the data and hide important features. Remove them.



df_2orMoreInj=df[df["number_of_persons_injured"]>2]

lat=df_2orMoreInj["latitude"]

lon=df_2orMoreInj["longitude"]

num_injured=df_2orMoreInj["number_of_persons_injured"]

num_killed=df_2orMoreInj["number_of_persons_killed"]



#print('number of people injured:')

#print(num_injured.min(), num_injured.max() )



temp=np.zeros(len(lat))+2



#Make scattered plot for num of injured people

"""

fig, ax = plt.subplots(figsize=(8,8))

im=ax.scatter(lat, lon, s=temp, c=num_injured, alpha=1,\

              label="persons injured", cmap="inferno_r",\

              vmax=8)

fig.colorbar(im, ax=ax)

plt.xlim(40.45,40.95)

plt.ylim(-74.30,-73.7)

plt.legend()

plt.show()

"""
#Repeat geographical visualization with plotly and mapbox

import plotly.plotly as py

import plotly.graph_objs as go

from datetime import date

today = str(date.today())



# Mapbox token

mapbox_access_token = 'pk.eyJ1IjoicHJpeWFyb3kiLCJhIjoiY2pxYmd2Z3dqM3g3eDN5czdhYXVoOXFxOCJ9.dLu9fZLObC1APYdddENU2A'

# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



data = [

    go.Scattermapbox(

    lat=df_2orMoreInj["latitude"],

    lon=df_2orMoreInj["longitude"],

    mode='markers',

    marker=dict(

        size=4,

        color=num_injured,      # set color to an array/list of desired values

        colorscale='Viridis',  #Choose a colorscale

        cmin=2,

        cmax=8,

        showscale=True,

        reversescale=True

    ),

)

]



layout = go.Layout(

    autosize=True,

    hovermode='closest',

    title = 'Number of Persons Injured from Collisions (All Years Combined), Updated '+today,

    mapbox= dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=40.72,

            lon=-74.0

        ),

        pitch=0,

        zoom=8.8

    ),

)

# actually show our figure

fig = dict( data=data, layout=layout )

iplot( fig, filename='NYPD_Collisions_Mapbox')
#Geographically visualize data with basemap and scattered plot

from mpl_toolkits.basemap import Basemap



#Draw map background

fig = plt.figure(figsize=(10, 10))



m = Basemap(resolution = 'h', projection = 'cyl',\

            llcrnrlon = -74.25,\

            urcrnrlon =  -73.7,\

            llcrnrlat = 40.49,\

            urcrnrlat = 40.91)

m.arcgisimage(service = "World_Shaded_Relief", xpixels = 1000)



#m.shadedrelief()

m.drawcoastlines(color='gray')

#m.drawcounties()

m.drawstates(color='gray')



# 2. scatter city data, with color reflecting num people injured

x, y = m(lon,lat)

m.scatter(x, y, s=temp, c=num_injured, alpha=0.5,\

          cmap="inferno_r",vmax=8)



# 3. create colorbar and legend

plt.colorbar(label=r'persons injured')

plt.clim(2, 8)

#plt.xlim(40.45,40.95)

#plt.ylim(-74.30,-73.7)

#plt.legend()

#plt.show()
####################################################################

#Plot trend of collisions, injured people in Manhattan area

####################################################################

import datetime

#Check time range in our data

#print("min date, max date in our data are:")

#print(df["date"].min(), df["date"].max())



#We see that we have data for last six years

#Extract year from date and add to the dataframe

df['year']=pd.DatetimeIndex(df['date']).year



#Convert zipcode from string to int

df.zip_code=pd.to_numeric(df.zip_code, errors='coerce').fillna(0).astype(np.int64)

#Now zipcode type is int. This allows us to apply boolean conditions to zipcodes.

#Keep zipcodes from Manhattan area only

df_mh=df[(df["zip_code"]>10000) & (df["zip_code"]<10281)]



#Further divide the data into regions within Manhattan

df_whi=df_mh.loc[df_mh['zip_code'].isin([10031,10032,10033,10034,10040])] #Washigton Heights & Inwood 

df_h=df_mh.loc[df_mh['zip_code'].isin([10026,10027,10039,10037,10039,10029,10035])] #Harlem  

#Sanity check:

#print("data in Harlem. Zip code min, max are:")

#print(df_h["zip_code"].min(), df_h["zip_code"].max())

df_ews=df_mh.loc[df_mh['zip_code'].isin([10002,10003,10009,10021,10028,10044,\

                                         10065,10075,10128,10023,10024,10025])] #East Side, West Side

df_cmhgv=df_mh.loc[df_mh['zip_code'].isin([10001,10011,10018,10019,10020,\

                                         10036,10012,10013,10014,10010,\

                                         10016,10017,10022])] #Chelsey, Murray Hill, Greenwich Village

df_lm=df_mh.loc[df_mh['zip_code'].isin([10004,10005,10006,10007,10038,10280])] #Lower Manhattan



#Density plot to check collisions (all yrs combined) in manhattan area only

#fig_hex,ax_hex=plt.subplots(1,3,figsize=(21,5))

fig,ax = plt.subplots(1,1,figsize=(10, 10))

naruto=ax.hexbin(y=df_mh["latitude"],x=df_mh["longitude"],\

                      gridsize=50,cmap="cool",mincnt=100,\

                      extent=(-74.017,-73.91,40.70,40.875))

cbar=fig.colorbar(naruto, ax=ax)

cbar.ax.tick_params(labelsize=15)

ax.set_title("Collisions in Manhattan", fontweight="bold", fontsize=20)

ax.tick_params(axis="x", labelsize=14)

ax.tick_params(axis="y", labelsize=14)

ax.set_xlabel('Longitude', fontsize=15)

ax.set_ylabel('Latitude', fontsize=15)

#I am a Naruto fan, as you can see :)



plt.show()



#Plot trend of collisions with time in various regions within Manhattan

#fig3a, ax3a = plt.subplots(figsize=(8,5))

fig_hex,ax_hex=plt.subplots(1,2,figsize=(21,10))

df_whi.groupby('year')['zip_code'].count().plot(ax=ax_hex[0],linewidth=4, label='W.Heights & Inwood')

df_h.groupby('year')['zip_code'].count().plot(ax=ax_hex[0],linewidth=4, label='Harlem')

df_ews.groupby('year')['zip_code'].count().plot(ax=ax_hex[0],linewidth=4, label='East,West Side')

df_cmhgv.groupby('year')['zip_code'].count().plot(ax=ax_hex[0],linewidth=4, label='Chelsey, M.Hill, G.Village')

df_lm.groupby('year')['zip_code'].count().plot(ax=ax_hex[0],linewidth=4, label='Lower Manhattan', fontsize=15)

ax_hex[0].set_xlabel('Year', fontsize=15)

ax_hex[0].set_ylabel('Number of collisions', fontsize=15)

ax_hex[0].legend(fontsize=15, frameon=False)



# Find persons injured per 100 collisions per year & plot

#fig3, ax3 = plt.subplots(figsize=(8,5))

df_whi.groupby('year').apply(lambda x: x['number_of_persons_injured'].agg('mean')*100).plot(ax=ax_hex[1],linewidth=4, label='W.Heights & Inwood')

df_h.groupby('year').apply(lambda x: x['number_of_persons_injured'].agg('mean')*100).plot(ax=ax_hex[1],linewidth=4,label='Harlem')

df_ews.groupby('year').apply(lambda x: x['number_of_persons_injured'].agg('mean')*100).plot(ax=ax_hex[1],linewidth=4, label='East,West Side')

df_cmhgv.groupby('year').apply(lambda x: x['number_of_persons_injured'].agg('mean')*100).plot(ax=ax_hex[1],linewidth=4, label='Chelsey, M.Hill, G.Village')

df_lm.groupby('year').apply(lambda x: x['number_of_persons_injured'].agg('mean')*100).plot(ax=ax_hex[1],linewidth=4, label='Lower Manhattan', fontsize=15)

ax_hex[1].set_xlabel('Year', fontsize=15)

ax_hex[1].set_ylabel('Persons injured per 100 collisions', fontsize=15)

#ax_hex[1].legend(fontsize=15,frameon=False)



#Set main title for subplots

fig_hex.suptitle("Trends in Manhattan", fontweight="bold", fontsize=20)



plt.show()



# Conclusions:



# 1) The region encompassing Chelsey, M. Hill and G. Village has the maximum record of 

# collisions,as is reflected from the hexbin plot as well as the plot which shows the 

# number of collisions versus time. 



# 2) Between 2015 and 2016, the number of collisions dropped significantly in Manhattan,

# particularly in the East-West side and the Chelsey, M. Hill & G. Village region. 

# This indicates that perhaps strong measurements were taken by the city officials 

# and trafffic safety groups during this time. 

  

# 3) Average collisions recorded in Harlem is about 4 times lower than the Chelsey, 

# M. Hill and G.Village region. However, the collisions in Harlem see about 1.7 times

# more injuries per 100 collisions. 



# 4) The lower Manhattan area seems to be the safest when it comes to the number 

# of collisions per year. However, the injuries suffered per 100 collisions rose 

# by 7 persons from 2015 to 2017.     