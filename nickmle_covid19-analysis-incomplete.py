# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# data collection and data load and extractuib part will take places --#
# data extraction process is more likely be automatic that people can load the data from --> 
# there are some external process plus some random processes -- that can have several random processe
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from bs4 import BeautifulSoup
import requests
import sklearn 
import datetime
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# to affect the configuration of the data all the world have in order to build the better performing models --> 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Please don't use API_key, it is free trial version with  500 maximum perday.
# this should be enough only for this notebook.
# please get one for you
api_key = "0ebe4425fbad41d8aaf25354200504"

"""
basics statistical analysis
forecasting of the general trends on cases
Main Ideas:
    - Prediction of deaths based on the values of confirmed cases.
    - Speed of recovery <--> confirmed cases.
    - association between tempeture and COVID-19 spread
        - For this, I am open for suggestions --> since the very good historical temperature data provider api is key to accomplish.
        
Dashboard build:

"""
"""
Project goal to study to about the  association between tempeture and COVID-19 spread


"""


confirmed =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
deaths =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
recovered =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
#there are several goals to achieve in the conservations those are coming to with the actions that is taking shape -->  
print(confirmed.shape, deaths.shape, recovered.shape)#  
confirmed.head()
len(confirmed['Country/Region'].unique())# total of 178 countries have infected with the desease --> that can run in several different runs --> 


def getcoronatable():
    
    
    """
    There is somepanddemica that can used to change several items -- making tmore reports of the people 
    This function will get the data from https://www.worldometers.info/coronavirus/ sites -- and synchronize it in the terms --.
    
    managing the  loading cycle of the  table -- and update the analyze for the data --> running service. some works needs to be done .. with thoses
    
    this is the particular event point -- that program that is running the returning the dashboard that deals with basic information -->  
    There are some minimal elements for the cluster centers and belong with some more tasks with several step procedures --> 
    can use the training in and out --> that will have cluster with the shorter distance .. 
    there are syntatic kmeans -- that can 
    
    """
    response = requests.get("https://www.worldometers.info/coronavirus/")
    print(response.status_code)
    namepart = str(datetime.datetime.now())
    if response.status_code == 200:
        print("Successful load of the website.")# there is the successful load from the website -- will get printed --> 
    bs_content = BeautifulSoup(response.content, "html")
    
    data_in_html = bs_content.find_all("table")# this is the data file --# it is possible to do automatic synchronazation every 30 mins to load the data -->

    rows = data_in_html[0].find_all("tr")# all the features those are columns -
    #bs_content bs contents will be loaded in the many frames  that can run on other platforms --
    columns = []
    for feature in rows[0].text.split("\n"):
        if feature!= "":
            columns.append(feature)
    #data = pd.DataFrame(columns = columns)
    #data.head()
    print(columns, len(columns))
    data_rows=  []
    rows = rows[1:]
    # there could be some values those are hard to extract valuable data from --> needs have particular runs -- that cannot change the values -- 
    
    # there are 13 columns passed but --> 11 columns has been created as the data frame -->  
    
    # 
    print("Can run also can't run the data -- mapping stages") 
    
    for row in rows:
        row = row.text.split("\n")[1:]
        row = row[:-1]# there is also nonetype elements that needs to be downloaded to run on the rest of the cases .. that can run --> higher performance algorithm --> 
        date_data = row[-1]
        
        row = row[:-2]
        row.append(date_data) # if does the extend comes into the version then it is more likely that  --> there could last string that is converted in the  list --by each character
        
        if len(row) == len(columns):
            
            data_rows.append(row)
    print(len(data_rows))
    # this  live -->  coding from --
    print("The length of the each row that is extracted: ", len(data_rows[0]))
    print("Data inside the first row is : ", data_rows[0])
    print("Data inside the second row is : ", data_rows[1])
    print("Data inside the third row is : ", data_rows[2])
    data = pd.DataFrame(data_rows, columns = columns)
    data.to_csv("data"+namepart+".csv", index = False, encoding = "utf-8")# 
    # encode the data by  utf-8 code and run  rest of the model -- that can run --.
    return data
# there could be several runs
# values needs to be run on different services --> 
# CORR VALUE table creation is important right -- with several different activation functions ... 
# vals and running examples -- that can be changed --> in several dimensional matrix, and there can have several dimensional matrix --> very wide matrix -->  
# go two vectors and stack them together --> 
# exactly having in the neural network with several elements -- that can run with more complex models --> 
# can it be worth to spend even though I can earn the amound I want to earn????  this 
# is the question 
# there could be some runs needs to deployed 
# there could be some run that can taken -->  load the rest --> 
data = getcoronatable()#
# some runs that
# 

# there are different cases and different general values also some conditions that can be shown and derived from the general approaches --> fixing the general values --> 
# there can be several values -->  very basica and general -- use cases -->  
index = data["Country,Other"].values
print(len(index), data.shape)# 20 items as the index of the data  frame that is going to run --> 
data.head()
# interactive dahsboards build with bokeh python
# 
from bokeh.io import show
from bokeh.models.grids import Grid
from bokeh.models.plots import Plot
from bokeh.models.axes import LinearAxis
from bokeh.models.ranges import Range1d
from bokeh.models.glyphs import Line, Cross
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tickers import SingleIntervalTicker, YearsTicker
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.annotations import Title, Legend, LegendItem
# every important steps that will take seriously int the cases -- that can be significant --> building the requ

confirmed.tail()# There could be several runs -- those are very critical to run the better performing models -- 
confirmed[confirmed["Country/Region"]  == "US"]#in some countries there state data included and for some there are no state or sub regional data included.
print("Total number of geographical regional data collected: ", confirmed.shape)
confirmed.head(2)# sample rows from the top
confirmed.tail(2)# sample rows from bottom 
def get_total(cols):
    
    """
    cols has two values inside it.
    1st -> Country/Region column
    2nd -> 3/31/20 that particular days data point.
    
    for some regions there are sub regional data points. so for particular day in order to get the total need to sum
    filtered values in the column
    
    """
    confirmed

cols = ['Country/Region','3/31/20']
confirmed[confirmed['Country/Region'] == "China"][cols].sum()
plt.show()# with no address == directly show th graph --> 
# first element and the last element from the row --> 
#rows = rows[1:]# first element row of the data will be removed since there is the  header of the columns 
deaths[deaths['Country/Region'] == "China"][cols].plot()
plt.show()#
cols = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',
       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',
       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',
       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',
       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',
       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',
       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',
       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',
       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20',
       '3/21/20', '3/22/20']
current_deaths = deaths[deaths['Country/Region'] == "China"][cols].sum()
current_deaths.plot()
plt.show()
import pandas as pd
import numpy as np
import os
# there are some risk managements --> that can work with several sectors -- that will work out with 
# potentially high observeable results --> that can be changed --> 
# there was an entry or no entry --> to go in t that country -->needs to be update with several values- 
# meret similar clusters -- and convergence  of the clusters --> bottom up or greedy algorithsms --> that can be update -- several values that is going to show the the final objective --> 
os.listdir("../input/novel-corona-virus-2019-dataset/")# need to build currrent version of data  set that will not be updated with several values or that can be changed withs everal values-
# that needs to be updated with several options that is not going to work with al
# all the dedicated version of the software that is currently working on with --> 
# can deploy dl model to run on those -->  
confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

print(confirmed.shape, recovered.shape, deaths.shape)
confirmed.columns# information inside the state that will be load 
# geographics -- mapping
# 
dates = ['1/22/20', '1/23/20',
       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',
       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',
       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',
       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',
       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',
       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',
       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',
       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',
       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20',
       '3/21/20', '3/22/20', '3/23/20', '3/24/20', '3/25/20', '3/26/20']
geo_data = ['Province/State', 'Country/Region', 'Lat', 'Long']
plt.plot(confirmed[dates].sum().values)
plt.legend()
plt.show()
# rate function
rates = [0]# supposing that starts from 100 from first day of the year 
daily_change = [100]
series = confirmed[dates].sum().values
for i in range(1, len(confirmed[dates].sum().values)):
    current = (series[i]-series[i-1])/series[i]
    temp = series[i]-series[i-1]
    rates.append(current)
    daily_change.append(temp)
plt.figure(figsize = (15, 10))#need to do grid enabler in the dataset -- visualization 
plt.plot(rates)
plt.show()# we can define that the growth precential rate  is still exponent

plt.figure(figsize=(15, 10))
plt.plot(daily_change)
plt.xlabel("Day started from January 22nd 2020")
plt.ylabel("Number of impected people daily")
plt.legend()
plt.show()

"""
There has to be triple measure -- 
On comfirmed cases
On Recovered cases
On deaths

"""
series[:5]
for i in confirmed['Country/Region'].values:
    print(i)
data1 = pd.DataFrame(columns = confirmed.columns)
data1 = data1.append(confirmed.iloc[0], ignore_index = True)

data1
data1 = data1.drop(index = 0)

data= confirmed
data[data["Lat"]!=33.0]

import matplotlib.pyplot as plt

#'%.3f'%geo_point0[0]# wuth 3 data

def get_geomapping(geo_arr):
    # this will get the particular geolocation array with lenght of 2 
    lat = geo_arr[0] # 
    lon = geo_arr[1] # 
    lat = '%.3f'%lat
    lon = '%.3f'%lon
    return lat+","+lon
def lat_lon_str(confirmed):
    lat_lon = []
    for i in range(confirmed.shape[0]):
        geo_point = confirmed[["Lat", "Long"]].iloc[i].values
        lat_lon.append(get_geomapping(geo_point))
    return lat_lon
lat_lon = lat_lon_str(confirmed)
    
lat_lon[:10]
from bs4 import BeautifulSoup
import json
import requests
import datetime

print("webpage renderer loaded!")
#today

def  get_weather_data(particular_lat_lon):
    api_key = "0ebe4425fbad41d8aaf25354200504"
    today = str(datetime.datetime.today().date())
    start = str(datetime.datetime.today().date()-datetime.timedelta(35)) # try to load the last 30 days of weather data --> 
    print("Start date : ", start, "End date : ", today)
    # I can get he more data points but, the limit is that historical data will return 5 weeks of the full data points with 24hrs  of temperature
    """
    start is the start date
    today is the enddate
    difference between today  and start date is 30 days with no more time and seconds delta by timedelta element
    particular_lat_lon is the  latitude and longitude.
    api_key is provided api_key from the platform that is using to get the weather data. 
    
    
    """
    
    # on q key in the dictionary it has to be modified based on the values it is getting from the outside data points
    params = {"key":api_key, "q":particular_lat_lon, "format":"json", "tp":24, "date":start, "enddate":today}

    print(params) # the filter the data from  here to load the it is possible  to make star t and end points.
    # 
    response = requests.get("https://api.worldweatheronline.com/premium/v1/past-weather.ashx", params = params)
    historical_weather = json.loads(response.content)
    dates = []
    maxtempC = []
    mintempC = []
    avgtempC = []
    daylighthours = []
    totalSnow = [] # basically the rain or snow that will drop  when it is 
    df = pd.DataFrame(columns = ["date", "maxtempC", "mintempC", "avgtempC", "sunHour", "totalSnow_cm"])
    for weather in historical_weather["data"]["weather"]:
        dates.append(weather['date'])
        maxtempC.append(weather['maxtempC'])
        mintempC.append(weather["mintempC"])
        avgtempC.append(weather["avgtempC"])
        daylighthours.append(weather["sunHour"])
        totalSnow.append(weather["totalSnow_cm"])
    df["date"] = dates
    df["maxtempC"] = maxtempC
    df["mintempC"] = mintempC
    df["avgtempC"] = avgtempC
    df["sunHour"] = daylighthours
    df["totalSnow_cm"] = totalSnow
    
    return df
start = str(datetime.datetime.today().date()-datetime.timedelta(35)) # try to load the last 30 days of weather data --> 
start
ll = lat_lon[0]
df = get_weather_data(ll)
df.tail()
# it is starts from where it needs to be started -->
df.shape  # significant -->
confirmed.head()
#top_30 countries will be included in the study
cols = ["Country/Region", "4/3/20"]
col = confirmed.columns[-1]
# try to condside
print(col)
lastday = confirmed[confirmed['4/3/20'] > 1000][cols].sort_values("4/3/20", ascending = False)
len(lastday["Country/Region"].unique())# there are several regions in that are not unique -- since included sub regions  are in the data points
def build_pie(confirmed, pdaycol):
    """
    pdaycol is the name of the column that is getting updated when the data runs with the higher efficiency --> 
    
    """
    #pdaycol = '3/31/20'
    print(pdaycol, "the  CoronaPie Chart")# for some people showing this would
    # slicing  data by the pandemic epicenters
    #confirmed['3/31/20'].sum()# there are some pandemic related data extraction methods that are very prevalent to run for entire cases --> only for the serious cases it will be harder to easer to load and better to perform the machine learning methods --.
    # setting 5 percent cutline 
    cutline  = confirmed[pdaycol].sum()*.01
    # cutline countries with more 1000
    col = ["Country/Region", pdaycol]
    morethan10k = confirmed[confirmed[col][pdaycol]>cutline][col]


    rest_sum = confirmed[confirmed[col][pdaycol]<=cutline][col].sum().values[1]
    print(rest_sum)
    Countries = list(morethan10k["Country/Region"].values)
    Countries.append("Other")
    TotalCases = list(morethan10k[pdaycol])# pday is the particular day that frame that is going to be created in order to run in der
    TotalCases.append(rest_sum)
    data  = []
    for i in range(len(Countries)):
        temp = []
        temp.append(Countries[i])
        temp.append(TotalCases[i])
        data.append(temp)
    #data

    data = pd.DataFrame(data =data, columns = ["Countries", "p_day"]) # p_day is the particular day
    data = data.sort_values("p_day", ascending= False)
    explosion = []
    # try to explode countries with more than 10% percent of the total case in particary days has been exploded

    for i in data["p_day"].values:
        if i/data["p_day"].sum()> .1:
            explosion.append(0.1)
        else:
            explosion.append(0)
    print(explosion)
    plt.figure(figsize=(15, 15))
    plt.pie("p_day", labels = "Countries", explode = explosion, data = data, autopct = "%.0f%%")
    plt.title(pdaycol)
    plt.show()

    
build_pie(confirmed, '4/7/20')
def build_sevendays():
    last_sevendays = confirmed.columns[-7:]
    
    for day in last_sevendays:
        build_pie(confirmed[last_sevendays], day)
#build_sevendays()


confirmed['4/7/20'].sum()
last_sevendays = confirmed.columns[-7:]
    
#for day in last_sevendays:
#    build_pie(day)

fig, axes = plt.subplots(2, 8)
axes = axes.ravel()
for i, ax in enumerate(axes):
    ax.plot()
    print(i)
import seaborn as sns# seaborn plotting library for  --> # --> predicted -- point __. 
# fix the algorithm to run the model.
# 
import requests

# using open weather data - to create the data base with the several  load able points 
weather_url = "http://bulk.openweathermap.org/snapshot/weather_14.json.gz?appid=6c3b580b086824dc7ce8152b4aec2c67"
print(weather_url)
# there are som end points for the api key -->

# getting threads -- small useable and workable with whole plans that can work with those all the case -- 
response = requests.get(weather_url)
response.content# lockdown on state --> 
import tensorflow as tf
import pandas as pd
import numpy as np 

# for the confirmed cases >> summation of the values has to be performeddata
countries = confirmed["Country/Region"].unique()
time_series_cols = ['1/22/20', '1/23/20',
       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',
       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',
       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',
       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',
       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',
       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',
       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',
       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',
       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20',
       '3/21/20', '3/22/20', '3/23/20', '3/24/20', '3/25/20', '3/26/20',
       '3/27/20', '3/28/20', '3/29/20', '3/30/20', '3/31/20', '4/1/20',
       '4/2/20', '4/3/20']
totals = confirmed[time_series_cols].sum().values
plt.plot(totals, color = "red")
plt.legend("Upper left")# there legends has be special more likely
plt.xlabel("time points")
plt.ylabel("number of cases")
plt.show()


sums_each_country = []
for country in countries:
    sums_each_country.append(confirmed[confirmed["Country/Region"] == country][time_series_cols].sum().values)
    
all_df = pd.DataFrame(data = sums_each_country,  columns = time_series_cols)
all_df.shape
all_df["Country/Region"]  = countries
all_df.shape
all_df[all_df['4/3/20']>1000].sort_values('4/3/20', ascending = False)
# there are several frames
# need to create the 
#import seaborn as sns

#sns.set(style = "ticks")
#sns.relplot("")
build_pie(all_df, '4/3/20')# now it is going into the perfect,  that can be loaded --> 

import matplotlib# not only a single plot that needs to build there are several data points
# there can be particular countries with data speculations -->
def get_ts_from_str(time_series_cols):
    time_series_cols_dt = []
    for i in time_series_cols:
        time_series_cols_dt.append(datetime.datetime.strptime(i, "%m/%d/%y"))
    len(time_series_cols_dt)
    return time_series_cols_dt

def build_top_profiles(all_df, pday):
    # there are several acceptable loads for the data 
    #pday = "4/3/20"  this particular date is thrown for the building of te graph
    
    all_df = all_df.sort_values(pday, ascending = False)
    
    top_countries = all_df["Country/Region"].values[:10]
    particular_country = all_df[all_df["Country/Region"] == top_countries[0]][time_series_cols].values 
    new_particular_country1 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[1]][time_series_cols].values 
    new_particular_country2 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[2]][time_series_cols].values 
    new_particular_country3 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[3]][time_series_cols].values 
    new_particular_country4 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[4]][time_series_cols].values 
    new_particular_country5 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[5]][time_series_cols].values 
    new_particular_country6 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[6]][time_series_cols].values 
    new_particular_country7 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[7]][time_series_cols].values 
    new_particular_country8 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[8]][time_series_cols].values 
    new_particular_country9 = np.reshape(particular_country,  (particular_country.shape[1],))
    particular_country = all_df[all_df["Country/Region"] == top_countries[9]][time_series_cols].values 
    new_particular_country10 = np.reshape(particular_country,  (particular_country.shape[1],))
    ts_cols = get_ts_from_str(time_series_cols)
    x_label_display = []
    for i in range(len(time_series_cols)):
        if  i%5 == 0:
            x_label_display.append(time_series_cols[i])
        else:
            x_label_display.append("")
    if x_label_display[-1] == "":
        x_label_display = x_label_display[:-1]
        x_label_display.append(time_series_cols[-1])
    print(len(x_label_display))
    plt.figure(figsize=(10,15))
    gs = matplotlib.gridspec.GridSpec(5, 5)
    #plt.title("Total")
    # there is country mapping that is taking place
    top1 = top_countries[0]
    top2 = top_countries[1]
    top3 = top_countries[2]
    top4 = top_countries[3]
    top5 = top_countries[4]
    top6 = top_countries[5]
    top7 = top_countries[6]
    top8 = top_countries[7]
    top9 = top_countries[8]
    top10 = top_countries[9]
    
    ax1 = plt.subplot(gs[:4, :3], title = top1)# up to 3 0,1, 2 x 0, 1, 2
    ax2 = plt.subplot(gs[0, 3:], title = top2) # last column by the graph
    ax3 = plt.subplot(gs[1, 3:], title = top3) # 
    ax4 = plt.subplot(gs[2, 3:], title = top4)
    ax5 = plt.subplot(gs[3, 3:], title = top5)
    ax6 = plt.subplot(gs[4, 0], title = top6)
    ax7 = plt.subplot(gs[4, 1], title = top7)
    ax8 = plt.subplot(gs[4, 2], title = top8)
    ax9 = plt.subplot(gs[4, 3], title = top9)
    ax10 = plt.subplot(gs[4, 4], title = top10)
    tempdf =  pd.DataFrame(columns = ["cases"])
    tempdf["cases"] = new_particular_country1
    tempdf.index = ts_cols
    tempdf = tempdf.to_period(freq ="W")
    print(tempdf.shape)
    ax1.plot(new_particular_country1)
    ax2.plot(new_particular_country2)
    ax3.plot(new_particular_country3)
    ax4.plot(new_particular_country4)
    ax5.plot(new_particular_country5)
    ax6.plot( new_particular_country6)
    ax7.plot( new_particular_country7)
    ax8.plot(new_particular_country8)
    ax9.plot(new_particular_country9)
    ax10.plot(new_particular_country10)
    #plt.title("Top 5 profiles in number of cases")
    plt.show()
    
    top_profiles = [new_particular_country1, new_particular_country2, new_particular_country3, new_particular_country4, new_particular_country5, new_particular_country6, new_particular_country7, new_particular_country8, new_particular_country9, new_particular_country10]
    daily_ups_tp = [] # daily ups for the top profiles--> 
    for current_list in top_profiles:
        temp = []
        index = 1
        for value in current_list[:-1]:
            temp.append(current_list[index]-value)
            index+=1
        daily_ups_tp.append(temp)
    # there is an inside variable no --> s global --> 
    plt.figure(figsize= (10, 15))
    gs = matplotlib.gridspec.GridSpec(5, 5)# two graphs that has to get build 
    #plt.title("Daily")
    ax1 = plt.subplot(gs[:4, :3], title = top1)# up to 3 0,1, 2 x 0, 1, 2
    ax2 = plt.subplot(gs[0, 3:], title = top2) # last column by the graph
    ax3 = plt.subplot(gs[1, 3:], title = top3) # 
    ax4 = plt.subplot(gs[2, 3:], title = top4)
    ax5 = plt.subplot(gs[3, 3:], title = top5)
    # next5 profiles of the countries will be included here to run --> 
    ax6 = plt.subplot(gs[4, 0], title = top6)
    ax7 = plt.subplot(gs[4, 1], title = top7)
    ax8 = plt.subplot(gs[4, 2], title = top8)
    ax9 = plt.subplot(gs[4, 3], title = top9)
    ax10 = plt.subplot(gs[4, 4], title = top10)
    ax1.plot(daily_ups_tp[0])
    ax2.plot(daily_ups_tp[1])
    ax3.plot(daily_ups_tp[2])
    ax4.plot(daily_ups_tp[3])
    ax5.plot(daily_ups_tp[4])
    ax6.plot(daily_ups_tp[5])
    ax7.plot(daily_ups_tp[6])
    ax8.plot(daily_ups_tp[7])
    ax9.plot(daily_ups_tp[8])
    ax10.plot(daily_ups_tp[9])
    plt.show()
build_top_profiles(all_df, "4/3/20")
deaths.shape
deaths.head()
print("Trends on the death 1: general trend 2: daily new deaths")
total_deaths_by_country = []
for country in deaths["Country/Region"].unique():
    temp = deaths[deaths["Country/Region"] == country][time_series_cols].sum().values
    total_deaths_by_country.append(temp)
deaths_df = pd.DataFrame(total_deaths_by_country, columns = time_series_cols)
deaths_df["Country/Region"] = deaths["Country/Region"].unique()
deaths_df = deaths_df.sort_values("4/3/20", ascending = False)
build_top_profiles(deaths_df, time_series_cols[-1])# build the last 
exdate = datetime.datetime.strptime(time_series_cols[0], "%m/%d/%y")

#time_series_cols_dt

print("Trends on the death 1: general trend 2: daily new deaths")
total_recovered_by_country = []
col = time_series_cols[-1]
for country in recovered["Country/Region"].unique():
    temp = recovered[recovered["Country/Region"] == country][time_series_cols].sum().values
    total_recovered_by_country.append(temp)
recovered_df = pd.DataFrame(total_recovered_by_country, columns = time_series_cols)
recovered_df["Country/Region"] = recovered["Country/Region"].unique()
recovered_df = recovered_df.sort_values(col, ascending = False)
build_top_profiles(recovered_df, time_series_cols[-1])# build the last 


