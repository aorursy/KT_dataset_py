import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from glob import glob as gb
#list all the directories

dirs=os.listdir("../input/zomato_data/")

dirs
len(dirs)
#storing all the files from every directory

li=[]

for dir1 in dirs:

    files=os.listdir(r"../input/zomato_data/"+dir1)

    #reading each file from list of files from previous step and creating pandas data fame    

    for file in files:

        

        df_file=pd.read_csv("../input/zomato_data/"+dir1+"/"+file,quotechar='"',delimiter="|")

#appending the dataframe into a list

        li.append(df_file.values)
len(li)
#numpys vstack method to append all the datafames to stack the sequence of input vertically to make a single array

df_np=np.vstack(li)
#no of rows is represents the total no restaurants ,now of coloumns(12) is columns for the dataframe

df_np.shape
#creating final dataframe from the numpy array

df_final=pd.DataFrame(df_np)
#adding the header columns

df_final=pd.DataFrame(df_final.values, columns =["NAME","PRICE","CUSINE_CATEGORY","CITY","REGION","URL","PAGE NO","CUSINE TYPE","TIMING","RATING_TYPE","RATING","VOTES"])
#displaying the dataframe

df_final
#header column "PAGE NO" is not required ,i used it while scraping the data from zomato to do some sort of validation,lets remove the column

df_final.drop(columns=["PAGE NO"],axis=1,inplace=True)
#display the dataframe again

df_final
#lets count how many unique cities are there 



df_final["CITY"].unique()
# import json and requests library to use googl apis to get the longitude ant latituide values

import requests

import json



#creating a separate array with all city names as elements of array

city_name=df_final["CITY"].unique()

li1=[]



#googlemap api calling url 

geo_s ='https://maps.googleapis.com/maps/api/geocode/json'

#iterating through a for loop for each city names 

for i in range(len(city_name)):



#i have used my own google map api, please use ypur own api     

 param = {'address': city_name[i], 'key': 'AIzaSyD-kYTK-8FQGueJqA2028t2YHbUX96V0vk'}

 

 response = requests.get(geo_s, params=param)

 

 response=response.text



 data=json.loads(response)



#setting up the variable with corresponding city longitude and latitude

 lat=data["results"][0]["geometry"]["location"]["lat"]

 lng=data["results"][0]["geometry"]["location"]["lng"]



#creating a new data frame with city , latitude and longitude as columns

 df2=pd.DataFrame([[city_name[i],lat,lng]])

 li1.append(df2.values)
#numpys vstack method to append all the datafames to stack the sequence of input vertically to make a single array

df_np=np.vstack(li1)
#creating a second dataframe with city name, latitude and longitude

df_sec=pd.DataFrame(df_np,columns=["CITY","lat","lng"])
#display the second dataframe contents

df_sec
#merge this data frame to the existing df_final data frame using merge and join features from pandas,and creating a new data frame

df_final2=df_final.merge(df_sec,on="CITY",how="left")
#display the contents , it will have longitude and latitude now

df_final2
#creating pandas series to hold the citynames and corresponding count of restuarnats in ascending order

li2=df_final["CITY"].value_counts().sort_values(ascending=True)
li2
#creating a empty dictionary

dc={}



#setting dictionary values as city name , count of restuarnat and key will city names as well

for i,j in li2.items():

    x=i + "," +str(j)

    dc.update({i:[i,j]})
#displaying the dictionary

dc
#creating another data frame from the above dictionary

df_map=pd.DataFrame.from_dict(dc,orient="index",columns=["CITY","COUNT"])
#displaying the data frame

df_map
#merging this data frame with df_sec data frame(which we created using city names,longitude and latitude)

df_map_final=df_map.merge(df_sec,on="CITY",how="left")
#displaying the new data frame this frame will be used for map ploting

df_map_final
#importing the libraries for map ploting

from matplotlib import cm

from matplotlib.dates import date2num

from mpl_toolkits.basemap import Basemap



# for date and time processing

import datetime
#lets take one data frame for top 20 cities with most retaurants counts 

df_plot_top=df_map_final.tail(20)
#displaying the data frame

df_plot_top
#lets plot this inside the map corresponding to the cities exact co-ordinates which we received from google api 

#plt.subplots(figsize=(20,50))

plt.figure(figsize=(50,60))

map=Basemap(width=120000,height=900000,projection="lcc",resolution="l",llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawcountries()

map.drawmapboundary(color='#f2f2f2')



map.drawcoastlines()







lg=np.array(df_plot_top["lng"])

lat=np.array(df_plot_top["lat"])

pt=np.array(df_plot_top["COUNT"])

city_name=np.array(df_plot_top["CITY"])



x,y=map(lg,lat)



#using lambda function to create different sizes of marker as per thecount 



p_s=df_plot_top["COUNT"].apply(lambda x: int(x)/2)



#plt.scatter takes logitude ,latitude, marker size,shape,and color as parameter in the below , in this plot marker color is always blue.

plt.scatter(x,y,s=p_s,marker="o",c='BLUE')

plt.title("TOP 20 INDIAN CITIES RESTAURANT COUNTS PLOT AS PER ZOMATO",fontsize=30,color='RED')
#lets plot this inside the map corresponding to the cities exact co-ordinates which we received from google api ,here marker color will be different as per marker size

#plt.subplots(figsize=(20,50))

plt.figure(figsize=(50,60))

map=Basemap(width=120000,height=900000,projection="lcc",resolution="l",llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawcountries()

map.drawmapboundary(color='#f2f2f2')



map.drawcoastlines()







lg=np.array(df_plot_top["lng"])

lat=np.array(df_plot_top["lat"])

pt=np.array(df_plot_top["COUNT"])

city_name=np.array(df_plot_top["CITY"])



x,y=map(lg,lat)



#using lambda function to create different sizes of marker as per thecount 



p_s=df_plot_top["COUNT"].apply(lambda x: int(x)/2)



#plt.scatter takes logitude ,latitude, marker size,shape,and color as parameter in the below , in this plot marker color is different.

plt.scatter(x,y,s=p_s,marker="o",c=p_s)

plt.title("TOP 20 INDIAN CITIES RESTAURANT COUNTS PLOT AS PER ZOMATO",fontsize=30,color='RED')
#lets plot with the city names inside the map corresponding to the cities exact co-ordinates which we received from google api ,here marker color will be different as per marker size

#plt.subplots(figsize=(20,50))

plt.figure(figsize=(50,60))

map=Basemap(width=120000,height=900000,projection="lcc",resolution="l",llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawcountries()

map.drawmapboundary(color='#f2f2f2')



map.drawcoastlines()







lg=np.array(df_plot_top["lng"])

lat=np.array(df_plot_top["lat"])

pt=np.array(df_plot_top["COUNT"])

city_name=np.array(df_plot_top["CITY"])



x,y=map(lg,lat)



#using lambda function to create different sizes of marker as per thecount 



p_s=df_plot_top["COUNT"].apply(lambda x: int(x)/2)



#plt.scatter takes logitude ,latitude, marker size,shape,and color as parameter in the below , in this plot marker color is different.

plt.scatter(x,y,s=p_s,marker="o",c=p_s)



for a,b ,c,d in zip(x,y,city_name,pt):

    #plt.text takes x position , y position ,text ,font size and color as arguments

    plt.text(a,b,c,fontsize=30,color="r")

   

    

    

plt.title("TOP 20 INDIAN CITIES RESTAURANT COUNTS PLOT AS PER ZOMATO",fontsize=30,color='RED')
#lets plot with the city names and restaurants count inside the map corresponding to the cities exact co-ordinates which we received from google api ,here marker color will be different as per marker size

#plt.subplots(figsize=(20,50))

plt.figure(figsize=(50,60))

map=Basemap(width=120000,height=900000,projection="lcc",resolution="l",llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawcountries()

map.drawmapboundary(color='#f2f2f2')



map.drawcoastlines()







lg=np.array(df_plot_top["lng"])

lat=np.array(df_plot_top["lat"])

pt=np.array(df_plot_top["COUNT"])

city_name=np.array(df_plot_top["CITY"])



x,y=map(lg,lat)



#using lambda function to create different sizes of marker as per thecount 



p_s=df_plot_top["COUNT"].apply(lambda x: int(x)/2)



#plt.scatter takes logitude ,latitude, marker size,shape,and color as parameter in the below , in this plot marker color is different.

plt.scatter(x,y,s=p_s,marker="o",c=p_s)



for a,b ,c,d in zip(x,y,city_name,pt):

    #plt.text takes x position , y position ,text(city name) ,font size and color as arguments

    plt.text(a,b,c,fontsize=30,color="r")

    #plt.text takes x position , y position ,text(restaurant counts) ,font size and color as arguments, like above . but only i have changed the x and y position to make it more clean and easier to read

    plt.text(a+60000,b+30000,d,fontsize=30)

   

    

    

plt.title("TOP 20 INDIAN CITIES RESTAURANT COUNTS PLOT AS PER ZOMATO",fontsize=30,color='RED')
#lets take one data frame for bottom 15 cities with minimum retaurants counts 

df_plot_bottom=df_map_final.head(15)
#displaying the data frame

df_plot_bottom
#lets plot with the city names and restaurants count inside the map corresponding to the cities exact co-ordinates which we received from google api ,here marker color will be different as per marker size

#plt.subplots(figsize=(20,50))

plt.figure(figsize=(50,50))

map=Basemap(width=120000,height=900000,projection="tmerc",resolution="l",llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=20,lon_0=88)

map.drawcountries()

map.drawmapboundary(color='#f2f2f2')



map.drawcoastlines()







lg=np.array(df_plot_bottom["lng"])

lat=np.array(df_plot_bottom["lat"])

pt=np.array(df_plot_bottom["COUNT"])

city_name=np.array(df_plot_bottom["CITY"])



x,y=map(lg,lat)



#using lambda function to create different sizes of marker as per thecount 



p_s=df_plot_bottom["COUNT"].apply(lambda x: int(x)*50)



#plt.scatter takes logitude ,latitude, marker size,shape,and color as parameter in the below , in this plot marker color is different.

plt.scatter(x,y,s=p_s,marker="o",c=p_s)



for a,b ,c,d in zip(x,y,city_name,pt):

    #plt.text takes x position , y position ,text(city name) ,font size and color as arguments

    plt.text(a-3000,b,c,fontsize=30,color="r")

    #plt.text takes x position , y position ,text(restaurant counts) ,font size and color as arguments, like above . but only i have changed the x and y position to make it more clean and easier to read

    plt.text(a+60000,b+30000,d,fontsize=30)

   

    

    

plt.title("BOTTOM 15 INDIAN CITIES MINIMUM RESTAURANT COUNTS PLOT AS PER ZOMATO",fontsize=30,color='RED')