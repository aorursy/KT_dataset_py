#import the modules

import os

import numpy as np

import pandas as pd

import plotly as plt 

from plotly.subplots import make_subplots

import plotly.graph_objects as go

#read all the files using pandas' read_csv

plant1_pg = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant2_pg = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")

plant1_ws = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

plant2_ws = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")



#creating a map

files = {

    0: plant1_pg,

    1: plant1_ws,

    2: plant2_pg,

    3: plant2_ws,

}
#plant-1 power generation data

files[0].sample(5)
#plant-1 weather sensor data

files[1].sample(5)
#converting the date-time in the right format using to_datetime 

#droping the plant_id column from all the files

for i in range(len(files)):

    files[i]["DATE_TIME"] = pd.to_datetime(files[i]["DATE_TIME"])

    files[i] =  files[i].drop(columns=["PLANT_ID"], axis=1) 

    
import plotly.express as px



def LineChart(temp_df,columns,start_date_time,end_date_time, title):

    temp_df = temp_df.loc[start_date_time : end_date_time]

    fig = px.line(temp_df[columns])

    fig.update_layout(title_text = title, title_x=0.5)

    fig.show()    
data1 = files[0][files[0].SOURCE_KEY == "3PZuoBAID5Wc2HD"]

data1= data1.set_index('DATE_TIME')



stime = "25-05-2020 05:00"

etime = "25-05-2020 20:00"

LineChart(data1, ["DC_POWER","AC_POWER"],stime,etime,"Power Generation during the Day")
data2 = files[1].set_index('DATE_TIME')



stime = "2020-05-25 05:00:00"

etime = "2020-05-25 20:00:00"

LineChart(data2,["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE"],stime,etime,"Temperature during the day")
data2 = files[1].set_index('DATE_TIME')

stime = "2020-05-25 05:00:00"

etime = "2020-05-25 20:00:00"

LineChart(data2,["IRRADIATION"],stime,etime,"Irradiation during the day")
#check for the data distribution and outliers

def BoxPlots(files, column1,column2,titles):

    fig = make_subplots(rows=2, cols=1, subplot_titles=titles)

    for i,file in enumerate(files):

        fig.add_trace(go.Box(x = list(file[column1].astype('int64')),name=column1),row=i+1,col=1)

        fig.add_trace(go.Box(x = list(file[column2].astype('int64')), name=column2),row=i+1,col=1)

    fig.update_layout(height=800, width=1000)

    fig.show()

    



BoxPlots([files[0],files[2]],"DC_POWER","AC_POWER",["Plant-1","Plant-2"])



BoxPlots([files[1],files[3]],"AMBIENT_TEMPERATURE","MODULE_TEMPERATURE",["Plant-1","Plant-2"])
tmp1 = files[0].copy()

tmp1["MONTH"] = tmp1["DATE_TIME"].dt.month

tmp1["YEAR"] = tmp1["DATE_TIME"].dt.year

plant1 = tmp1[tmp1.YEAR == 2020].sort_values('MONTH').groupby('MONTH').agg({"DAILY_YIELD":"sum"}).reset_index()

plant1.index.name = None



plant2 = pd.DataFrame(0, index=plant1.index, columns=['MONTH','DAILY_YIELD'])

plant2['MONTH'] = pd.DataFrame(range(1,13))



tmp2 = files[2].copy()

tmp2["MONTH"] = tmp2["DATE_TIME"].dt.month

tmp2["YEAR"] = tmp2["DATE_TIME"].dt.year 

tmp2 = tmp2[tmp2.YEAR == 2020].sort_values('MONTH').groupby('MONTH').agg({"DAILY_YIELD":"sum"}).reset_index()

tmp2.index.name = None

plant2.iloc[4:6] = tmp2.iloc[:].values
def BarMonth(temp_df,x,y, year, title):

    fig = px.bar(temp_df,x=x,y=y)

    fig.update_layout(title_text = title, title_x=0.5)

    fig.show()  
BarMonth(plant1,'MONTH','DAILY_YIELD',2020,"Plant-1 Monthly Yield")

BarMonth(plant2,'MONTH','DAILY_YIELD',2020,"Plant-2 Monthly Yield")