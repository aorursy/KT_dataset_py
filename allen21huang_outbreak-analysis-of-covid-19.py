import json

import folium

import webbrowser

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import folium.plugins as plugins

from folium.plugins import HeatMap
# load the json file of latitude and longtitude data

with open('/kaggle/input/region.json') as f: 

    region = json.load(f)
longitude = []

latitude = []

region_name = []
# get province,longitude,and latitude to create a daraframe of province's geographic data

# this part is for geographic visualization

longitude = []

latitude = []

region_name = []



for i in region['districts']:

    if i['level'] == 'province': 

        region_name.append(i['name'])

        longitude.append(i["center"]["longitude"])

        latitude.append(i["center"]["latitude"])



region_lookup = pd.DataFrame({'province':region_name,"longitude":longitude,"latitude":latitude})

region_lookup.head()
covidData = pd.read_csv("/kaggle/input/ncovdata.csv")
covidData.head()
# take a look at some information about this dataset

covidData.info()
covidData.dtypes
# convert the datatype of updateTime to datetime64, and get date from that column as a new column

def datimeTrans(dataset):

    dataset["updateTime"] = dataset["updateTime"].apply(pd.to_datetime)

    dataset["date"] = dataset["updateTime"].dt.date.apply(pd.to_datetime)

    return dataset
covidData = datimeTrans(covidData)
# here I just momentarily convert the datatype to string

covidData["date"] = covidData["date"].astype(str)
date = []
def deldup(dataset):    

    for i in range(len(dataset)):

        province = dataset.loc[i,"provinceName"]

        city = dataset.loc[i,"cityName"]

        if dataset.loc[i,"date"] not in date:

            date.append(dataset.loc[i,"date"])

            provinces = []

            citys = []

        if province not in provinces:

            provinces.append(province)

        if (province in provinces) & (city in citys):

            dataset = dataset.drop(i)

        if (province in provinces) & (city not in citys):

            citys.append(city)

    return dataset
covidData = deldup(covidData)
covidData.describe()
dayUnique = pd.DataFrame()

dayUnique = covidData.groupby(['provinceName', 'date'])['city_confirmedCount'].aggregate('sum').unstack()

dayUnique
def nullhandling(dataset):

    dataset.iloc[:,0].fillna(0,inplace = True)

    for j in range(len(dataset)):

        for i in range(len(dataset.columns)):

            if i == 0:

                continue

            elif dataset.isnull().iloc[j,i] == False:

                continue

            else:

                dataset.iloc[j,i] = dataset.iloc[j,i-1]

    return dataset
dayUnique = nullhandling(dayUnique)

dayUnique
dayUnique.style.background_gradient(cmap='gnuplot')
dayUnique_drophb = dayUnique.drop("湖北省")

dayUnique_drophb.style.background_gradient(cmap='twilight_shifted')
dayUnique_drophb = dayUnique_drophb.T.reset_index()

dayUnique = dayUnique.T.reset_index()
listofcolumns = []

for i in dayUnique_drophb.columns:

    if i != "date":

        listofcolumns.append(i)



dayUnique_drophb_melt = pd.melt(dayUnique_drophb, id_vars=['date'],

                                value_vars= listofcolumns, 

                                var_name='Province', value_name='Count')
listofcolumns = []

for i in dayUnique.columns:

    if i != "date":

        listofcolumns.append(i)



dayUnique_melt = pd.melt(dayUnique, id_vars=['date'],

                                value_vars= listofcolumns, 

                                var_name='Province', value_name='Count')
px.line(dayUnique_melt, x="date", y="Count",color ='Province',

        title='Outbreak in China from Jan 25 - Feb 15')
px.line(dayUnique_drophb_melt, x="date", y="Count",color ='Province',

        title='Outbreak in China (excluding Hubei Province) from Jan 25 - Feb 15')
fig = px.bar(dayUnique_melt, 

             y="date", x="Count", color='Province', orientation='h', height=700,

             title='Number of Confirmed in China')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')

fig.show()
fig = px.bar(dayUnique_drophb_melt, 

             y="date", x="Count", color='Province', orientation='h', height=700,

             title='Number of Confirmed in China (excluding Hubei Province)')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')

fig.show()
px.bar(dayUnique_melt,x="Count", y="Province", color='Province', 

       orientation='h', height=800,title='Outbreak in China',

       animation_frame="date",range_x = [0,65000])
px.bar(dayUnique_drophb_melt,x="Count", y="Province", color='Province', 

       orientation='h', height=800,title='Outbreak in China (excluding Hubei Province)',

       animation_frame="date",range_x=[0,1400])
px.treemap(dayUnique_drophb_melt.sort_values(by='Count', ascending=False).reset_index(drop=True), 

           path=["Province"], values="Count")
dayUnique = pd.DataFrame()

dayUnique = covidData.groupby(['provinceName', 'date'])['city_confirmedCount'].aggregate('sum').unstack()

dayUnique
dayUnique_wuhan = covidData[covidData["provinceName"] == "湖北省"]

dayUnique_wuhan = dayUnique_wuhan.groupby(['cityName', 'date'])['city_confirmedCount'].aggregate('sum').unstack()
# I have defined this function at the begining

dayUnique_wuhan = nullhandling(dayUnique_wuhan)

dayUnique_wuhan.style.background_gradient(cmap='rainbow')
dayUnique_wuhan = dayUnique_wuhan.T.reset_index()

listofcolumns = []

# get all of the column names, store in a list

for i in dayUnique_wuhan.columns:

    if i != "date":

        listofcolumns.append(i)



dayUnique_wuhan_melt = pd.melt(dayUnique_wuhan, id_vars=['date'],

                                value_vars= listofcolumns, 

                                var_name='Province', value_name='Count')
fig = px.bar(dayUnique_wuhan_melt, 

             y="date", x="Count", color='Province', orientation='h', height=700,

             title='Number of Confirmed in Hubei Province')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')

fig.show()
px.bar(dayUnique_wuhan_melt,x="Count", y="Province", color='Province', 

       orientation='h', height=800,title='Outbreak in Hubei Province',

       animation_frame="date",range_x = [0,38000])
px.treemap(dayUnique_wuhan_melt.sort_values(by='Count', ascending=False).reset_index(drop=True), 

           path=["Province"], values="Count")
dayUnique_melt_region = pd.merge(region_lookup,dayUnique_melt,

                                left_on="province",right_on="Province",how='outer')
# here, some province name doesn't match

dayUnique_melt_region.isnull().sum()
# let's find it

dayUnique_melt_region.head()
# let's find it

for i in list(set(dayUnique_melt["Province"])):

    if i in region_lookup["province"].values:

        pass

    else:

        print(i)
region_lookup['province'][0] = '澳门'
del dayUnique_melt_region

dayUnique_melt_region = pd.merge(region_lookup,dayUnique_melt,

                                left_on="province",right_on="Province",how='outer')

del dayUnique_melt_region["province"]
dayUnique_melt_region.isnull().sum()
dayUnique_melt_region = dayUnique_melt_region.dropna()

dayUnique_melt_region.dtypes
# select only 2020-02-14

# currently data of 02-15 is incomplete

dayUnique_melt_region["date"] = dayUnique_melt_region["date"].apply(pd.to_datetime)

dayUnique_melt_region_lastday = dayUnique_melt_region.query("date == '2020-02-14'")

dayUnique_melt_region_lastday.head()
covmap = folium.Map(location=[36, 105], zoom_start=4)



for lat, lon, value, name in zip(dayUnique_melt_region_lastday["latitude"], 

                                 dayUnique_melt_region_lastday['longitude'], 

                                 dayUnique_melt_region_lastday['Count'], 

                                 dayUnique_melt_region_lastday['Province']):

    folium.CircleMarker([lat, lon],

                        radius=20,

                        popup = ('Province: ' + str(name) + '<br>'

                        'Confrimed: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(covmap)





covmap
# convert data into lists that can be used by folium

num = dayUnique_melt_region_lastday.shape[0]

lat = np.array(dayUnique_melt_region_lastday["latitude"][0:num])

lon = np.array(dayUnique_melt_region_lastday['longitude'][0:num])

confrimed = np.array(dayUnique_melt_region_lastday['Count'][0:num])

mapdata = [[lat[i], lon[i], confrimed[i]] for i in range(num)]
heatmap = folium.Map(location=[38, 100], zoom_start=4)

HeatMap(mapdata).add_to(heatmap)
# visualize the spread of COVID-19 

heatmap
dayunique_dq = covidData.query("date == '2020-02-14'").groupby(['provinceName', 'date'])["city_curedCount","city_deadCount","city_confirmedCount"].aggregate('sum').unstack()
# create two new columns of cure rate and death rate

dayunique_dq["cure rate"] = dayunique_dq["city_curedCount"]/dayunique_dq["city_confirmedCount"]

dayunique_dq["death rate"] = dayunique_dq["city_deadCount"]/dayunique_dq["city_confirmedCount"]
dayunique_dq.drop(["city_curedCount","city_deadCount","city_confirmedCount"],axis=1,inplace=True)

dayunique_dq.reset_index(inplace=True)

dayunique_dq.head()
px.bar(dayunique_dq,x="provinceName",y="cure rate",

       color="provinceName",title='Cure rate of each province')
px.bar(dayunique_dq,x="provinceName",y="death rate",

       color="provinceName",title='Death rate of each province')
dayUnique_gr = dayUnique.T

dayUnique_gr.drop(["2020-01-24","2020-01-25","2020-01-26"],inplace=True)

dayUnique_gr.drop(["澳门","西藏自治区"],axis=1,inplace=True)
pd.set_option('display.max_columns', None)

dayUnique_gr.head(10)
dayUnique_gr_2 = dayUnique_gr.copy()
for i in range(len(dayUnique_gr)):

    for j in range(len(dayUnique_gr.columns)):

        if i == 0:

            pass

        else:

            now = dayUnique_gr.iloc[i,j]

            past = dayUnique_gr.iloc[i-1,j]

            dayUnique_gr_2.iloc[i,j] = (now-past)/past
dayUnique_gr_2.drop("2020-01-27",inplace=True)

dayUnique_gr_2.head()
dayUnique_gr_2 = dayUnique_gr_2.reset_index()
listofcolumns = []

for i in dayUnique_gr_2.columns:

    if i != "date":

        listofcolumns.append(i)



dayUnique_gr_2_melt = pd.melt(dayUnique_gr_2, id_vars=['date'],

                                value_vars= listofcolumns, 

                                var_name='ProvinceName')
dayUnique_gr_2_melt.head()
px.line(dayUnique_gr_2_melt, x="date", y="value",color ='ProvinceName',

        title='Growth rate of coronavirus in each province')
px.bar(dayUnique_gr_2_melt,x="value", y="ProvinceName", color='ProvinceName', 

       orientation='h', height=800,title='Growth rate of coronavirus in each province',

       animation_frame="date",range_x = [0,2])