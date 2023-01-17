# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import csv

import matplotlib.pylab as pylab

%matplotlib inline



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

import datetime
#tab = pd.read_csv("../input/country-pop-data/country_pop_data.csv")

con = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

con_US = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv")

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 16,

        }
con["Province/State"].fillna("*", inplace = True)

df = con

countries = [ "Australia","Canada", "China"]

china = df.loc[df['Country/Region'] == 'China']

aus = df.loc[df['Country/Region'] == 'Australia']

can = df.loc[df['Country/Region'] == 'Canada']

frames = [aus,can,china]

country_v =0

df1=df.drop(df.index[2:-1])

for each_country in countries:

    #print(country_v)

    state = "*"

    #print(each_country)

    df1.iloc[country_v,0]=state

    df1.iloc[country_v,1]=each_country

    frame = frames[country_v]

   

    length = len(frame.columns)

    #print(frame)

    #print(length,state,each_country)

    for y in range(length-4):

        

        z=y+4

        total = 0

       # print(frame.columns.values[z])

       

        for x in range(len(frame)):

            

            new = frame.iloc[x,z]

            

            total = total + new

           # print(new,total)

            df1.iloc[country_v,z]= total

    country_v+=1        

   # print(total)   
frames = [df1, con]

confirmed = pd.concat(frames)
doubling = pd.DataFrame(columns=['SNo','Province/State','Country/Region','Lat','Long','Confirmed','Doubling Rate'])

count=0

con = confirmed

for i in range(len(con)): 

        count+=1

        latest = con.iloc[i,-1]

        state = con.iloc[i,0] 

        country = con.iloc[i,1]

        lat =con.iloc[i,2]

        long = con.iloc[i,3]

        

        doub = latest/2

        z = len(con.columns)-1

        v=np.nan

        if latest > 25:

            for j in range(z):

          

                value = con.iloc[i,z-j]

             

                if value <= doub:

              

                    v=j

                    

                    break

        d = (state,country,v)        

            #doubling = pad.concat(state,country,v)

        doubling.loc[count] = [count,state,country,lat, long,latest,v]
last_date_in_df  = con.columns.values[-1]

print("Data goes up to: ",last_date_in_df) 
doub=doubling.loc[doubling["Doubling Rate"]<7]

doub_w=doubling.loc[doubling["Doubling Rate"]<10]

doub1 = doub.drop(columns="Lat")

doub1 = doub1.drop(columns="Long")

doub1 = doub1.drop(columns="SNo")

doub1 = doub1.sort_values(by ='Country/Region')

doub1['Province/State'].fillna('*',inplace=True)

print("AREAS OF FASTEST SPREAD (CASES>25)")

count = doub1.shape[0] 

doub1.head(count)
# Counties that are doubling on a world map

import folium

import math



doubling = doubling[doubling['Doubling Rate'] == doubling['Doubling Rate'].max()]

map = folium.Map(location=[10, 0], tiles = "cartodbpositron", zoom_start=3.0,max_zoom=6,min_zoom=2)

for i in range(0,len(doub_w)):

    folium.Circle(location=[doub_w.iloc[i,3],

                            doub_w.iloc[i,4]],

                           

                            tooltip = "<h5 style='text-align:center;font-weight: bold'>"+doub_w.iloc[i]['Country/Region']+"</h5>"+

                            "<div style='text-align:center;'>"+str(np.nan_to_num(doub_w.iloc[i]['Province/State']))+"</div>"+

                            "<hr style='margin:10px;'>"+

                            "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

                            "<li>Cases: "+str(doub_w.iloc[i,5])+"</li>"+

                             "<li>Doubling Rate: "+str(doub_w.iloc[i,-1])+"</li>"+

                            

                            "</ul>"

                            ,

                            #radius=(math.sqrt(doub.iloc[i,5])*4000 ),

                            radius=(int((np.log(doub_w.iloc[i,-1]+1.00001)))+0.2)*50000,

                            color='orange',

                            fill=True,

                            fill_color='orange').add_to(map)

map    
doubling_US = pd.DataFrame(columns=['SNo','Province/State','Admin Area','Lat','Long','Confirmed','Doubling Rate'])

count=0

for i in range(len(con_US)): 

        count+=1

        latest = con_US.iloc[i,-1]

        state = con_US.iloc[i,6] 

        admin = con_US.iloc[i,5] 

        country = con_US.iloc[i,7]

        lat =con_US.iloc[i,8]

        long = con_US.iloc[i,9]

        doub = latest/2

        z = len(con_US.columns)-1

        v=np.nan

        if latest > 25:

            for j in range(z):

          

                value = con_US.iloc[i,z-j]

             

                if value <= doub:

              

                    v=j

                    

                    break

        d = (state,country,v)        

           

        doubling_US.loc[count] = [count,state,admin,lat, long,latest,v]
doub_US=doubling_US.loc[doubling_US["Doubling Rate"]<4]

doub_USA=doubling_US.loc[doubling_US["Doubling Rate"]<5]

doub1_US = doub_US.drop(columns="Lat")

doub1_US = doub1_US.drop(columns="Long")

doub1_US = doub1_US.drop(columns="SNo")

doub1_US = doub1_US.sort_values(by ='Admin Area')

doub1_US['Province/State'].fillna('*',inplace=True)

print("AREAS OF FASTEST SPREAD (CASES > 25)")

count = doub1_US.shape[0] 

doub1_US.head(count)


#doub_USA=doubling_US.loc[doubling_US["Doubling Rate"]<5]

doubling = doubling[doubling['Doubling Rate'] == doubling['Doubling Rate'].max()]

map_USA = folium.Map(location=[31, -84], tiles = "cartodbpositron", zoom_start=4.0,max_zoom=7,min_zoom=2)

for i in range(0,len(doub_USA)):

    folium.Circle(location=[doub_USA.iloc[i,3],

                            doub_USA.iloc[i,4]],

                           

                            tooltip = "<h5 style='text-align:center;font-weight: bold'>"+doub_USA.iloc[i]['Admin Area']+"</h5>"+

                            "<div style='text-align:center;'>"+str(np.nan_to_num(doub_USA.iloc[i]['Province/State']))+"</div>"+

                            "<hr style='margin:10px;'>"+

                            "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

                            "<li>Cases: "+str(doub_USA.iloc[i,5])+"</li>"+

                            "<li>Doubling Rate: "+str(doub_USA.iloc[i,-1])+"</li>"+

                            

                            "</ul>"

                            ,

                            #radius=(math.sqrt(doub_USA.iloc[i,5])*4000 ),

                            radius=(int((np.log(doub_USA.iloc[i,-1]+1.00001)))+0.2)*50000,

                            color='red',

                            fill=True,

                            fill_color='red').add_to(map_USA)

map_USA