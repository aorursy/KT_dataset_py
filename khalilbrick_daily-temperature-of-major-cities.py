import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



!pip install plotly

!pip install chart_studio



import plotly.tools as tls

import plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from chart_studio import plotly as py

from plotly.offline import iplot



%matplotlib inline

df = pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv")

df.head()
len(df.Country.unique())
df.tail()
df.shape
df.info()
df = df.drop_duplicates()

df.shape
df.count()
for col in df.columns: # check missing values (Nan) in every column

    print("The " + col + " contains Nan" + ":" + str((df[col].isna().any())))
for col in df.columns: # check missing values (Zeros) in every column

    print("The " + col + " contains 0" + ":" + str((df[col] == 0 ).any()))

df = df[df.Day != 0]

df.head()
df = df[(df.Year!=200) & (df.Year!=201)]

df.head()
Average_Temperture_in_every_region = df.groupby("Region")["AvgTemperature"].mean().sort_values()[-1::-1]

Average_Temperture_in_every_region = Average_Temperture_in_every_region.rename({"South/Central America & Carribean":"South America","Australia/South Pacific":"Australia"})

Average_Temperture_in_every_region
plt.figure(figsize = (15,8))

plt.bar(Average_Temperture_in_every_region.index,Average_Temperture_in_every_region.values)

plt.xticks(rotation = 10,size = 15)

plt.yticks(size = 15)

plt.ylabel("Average_Temperture",size = 15)

plt.title("Average Temperture in every region",size = 20)

plt.show()
# change the index to date

datetime_series = pd.to_datetime(df[['Year','Month', 'Day']])

df['date'] = datetime_series

df = df.set_index('date')

df = df.drop(["Month","Day","Year"],axis = 1)

df.head()
region_year = ['Region', pd.Grouper(freq='Y')]

df_region = df.groupby(region_year).mean()

df_region.head()
plt.figure(figsize = (15,8))

for region in df["Region"].unique():



    plt.plot((df_region.loc[region]).index,df_region.loc[region]["AvgTemperature"],label = region) 

    

plt.legend()

plt.title("Growth of the average Temperture in every region over time",size = 20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.show()
df_earth = df.groupby([pd.Grouper(freq = "Y")]).mean()

df_earth.head()
plt.figure(figsize = (15,8))

plt.plot(df_earth.index,df_earth.values,marker ="o")

plt.xticks(size =15)

plt.ylabel("average Temperture",size = 15)

plt.yticks(size =15)

plt.title("Growth of the average Temperture (Earth)",size =20)

plt.show()
top_10_hotest_Cities_in_The_world = df.groupby("City").mean().sort_values(by = "AvgTemperature")[-1:-11:-1]

top_10_hotest_Cities_in_The_world
plt.figure(figsize = (15,8))

plt.barh(top_10_hotest_Cities_in_The_world.index,top_10_hotest_Cities_in_The_world.AvgTemperature)
city_year = ['City', pd.Grouper(freq='Y')]

df_city = df.groupby(city_year).mean()

df_city.head()
plt.figure(figsize = (20,8))

for city in top_10_hotest_Cities_in_The_world.index:

    plt.plot(df_city.loc[city].index,df_city.loc[city].AvgTemperature,label = city)

plt.legend()

plt.yticks(size = 15)

plt.xticks(size = 15)

plt.ylabel("Average Temperature",size = 15)

plt.title("The Growth of the Temperture in the hotest Cities in The world",size = 20)

plt.show()
hotest_Countries_in_The_world = df.groupby("Country").mean().sort_values(by = "AvgTemperature")

hotest_Countries_in_The_world.tail()
plt.figure(figsize = (20,8))

plt.bar(hotest_Countries_in_The_world.index[-1:-33:-1],hotest_Countries_in_The_world.AvgTemperature[-1:-33:-1])

plt.yticks(size = 15)

plt.ylabel("Avgerage Temperature",size = 15)

plt.xticks(rotation = 90,size = 12)

plt.title("The hotest Countries in The world",size = 20)

plt.show()
code = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv") # this is for the county codes

code= code.set_index("English short name lower case")

code.head()
code = code.rename(index = {"United States Of America":"US","Côte d'Ivoire":"Ivory Coast","Korea, Republic of (South Korea)":"South Korea","Netherlands":"The Netherlands","Syrian Arab Republic":"Syria","Myanmar":"Myanmar (Burma)","Korea, Democratic People's Republic of":"North Korea","Macedonia, the former Yugoslav Republic of":"Macedonia","Ecuador":"Equador","Tanzania, United Republic of":"Tanzania","Serbia":"Serbia-Montenegro"})

code.head()
hott = pd.merge(hotest_Countries_in_The_world,code,left_index = True , right_index = True , how = "left")

hott.head()
data = [dict(type = "choropleth",autocolorscale = False, locations=  hott["Alpha-3 code"], z = hott["AvgTemperature"] ,

              text = hott.index,colorscale = "reds",colorbar = dict(title = "Temperture"))]                         
layout = dict(title = "The Average Temperature around the world",geo = dict(scope = "world",projection = dict(type = "equirectangular"),showlakes = True,lakecolor = "rgb(66,165,245)",),)
fig = dict(data = data,layout=layout)

iplot(fig,filename = "d3-choropleth-map")
Variation_world = df.groupby(df.index.month).mean()

Variation_world = Variation_world.rename(index = {1:"January",2:"February" ,3:"March" ,4:"April" ,5:"May" ,6:"June" ,7:"July" ,8:"August" ,9:"September" ,10:"October" ,11:"November" ,12:"December" })
plt.figure(figsize=(18,8))

sns.barplot(x=Variation_world.index, y= 'AvgTemperature',data=Variation_world,palette='Set2')

plt.title('AVERAGE MEAN TEMPERATURE OF THE WORLD',size = 25)

plt.xticks(size = 15)

plt.yticks(size = 20)

plt.xlabel("Month",size = 20)

plt.ylabel("AVERAGE MEAN TEMPERATURE",size = 15)

plt.show()
Variation_UAE = df.loc[df["Country"] == "United Arab Emirates"].groupby(df.loc[df["Country"] == "United Arab Emirates"].index.month).mean()

Variation_UAE = Variation_UAE.rename(index = {1:"January",2:"February" ,3:"March" ,4:"April" ,5:"May" ,6:"June" ,7:"July" ,8:"August" ,9:"September" ,10:"October" ,11:"November" ,12:"December" })
plt.figure(figsize=(18,8))

sns.barplot(x=Variation_UAE.index, y= 'AvgTemperature',data=Variation_UAE,palette='Set2')

plt.title('Variation of the mean Temperature Over The 12 months in the United Arab Emirates',size = 20)

plt.xticks(size = 15)

plt.yticks(size = 20)

plt.xlabel("Month",size = 20)

plt.ylabel("AVERAGE MEAN TEMPERATURE",size = 15)

plt.show()
plt.figure(figsize=(30,55))

i= 1 # this is for the subplot

for region in df.Region.unique(): # this for loop make it easy to visualize every region with less code

    

    region_data =df[df['Region']==region]

    final_data= region_data.groupby(region_data.index.month).mean()['AvgTemperature'].sort_values(ascending=False)



    final_data = pd.DataFrame(final_data)

    final_data = final_data.sort_index()



    final_data = final_data.rename(index = {1:"January",2:"February" ,3:"March" ,4:"April" ,5:"May" ,6:"June" ,7:"July" ,8:"August" ,9:"September" ,10:"October" ,11:"November" ,12:"December" })

    plt.subplot(4,2,i)

    sns.barplot(x=final_data.index,y='AvgTemperature',data=final_data,palette='Paired')

    plt.title(region,size = 20)

    plt.xlabel(None)

    plt.xticks(rotation = 90,size = 18)

    plt.ylabel("Mean Temperature",size = 15)

    i+=1

Average_Temperature_USA = df.loc[df["Country"] == "US"].groupby("State").mean().drop(["Additional Territories"],axis = 0)

Average_Temperature_USA.head()

usa_codes = pd.read_csv('../input/usa-states-codes/csvData.csv')

usa_codes =usa_codes.set_index("State")

Average_Temperature_USA = pd.merge(Average_Temperature_USA,usa_codes,how = "left",right_index = True,left_index = True)

Average_Temperature_USA.head()
data_usa = [dict(type = "choropleth",autocolorscale = False, locations=  Average_Temperature_USA["Code"], z = Average_Temperature_USA["AvgTemperature"] ,

              locationmode="USA-states",

              text = Average_Temperature_USA.index,colorscale = "reds",colorbar = dict(title = "Temperture"))]                         

layout_usa = dict(title = "The Average Temperature in the USA states",geo = dict(scope = "usa",projection = dict(type = "albers usa"),showlakes = True,lakecolor = "rgb(66,165,245)",),)
fig_usa = dict(data = data_usa,layout=layout_usa)

iplot(fig_usa,filename = "d3-choropleth-map")
Temperature_USA_year = df.loc[df["Country"] == "US"].groupby(pd.Grouper(freq = "Y")).mean()

Temperature_USA_year.head()
plt.figure(figsize = (15,8))

sns.barplot(x = Temperature_USA_year.index.year,y = "AvgTemperature",data = Temperature_USA_year)

plt.yticks(size = 15)

plt.xticks(size = 15,rotation = 90)

plt.xlabel(None)

plt.ylabel("Avgerage Temperature",size = 15)

plt.title("Average Temperature in USA from 1995 to 2020",size = 20)

plt.show()