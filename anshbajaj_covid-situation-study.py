# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib

import folium

from folium import Marker

def embed_map(m,file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name,width='100%',height='500px')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/usa-hospital-beds/USA_Hospital_Beds.csv")

data2=pd.read_csv("../input/covid-19-cases/04-20-2020.csv")

data
beds=data.sort_values(by='STATE_NAME').reset_index()

beds=beds.fillna(int("0"))

beds=beds.drop("index",axis=1)

beds["TOTAL_BEDS"]=beds.apply(lambda x:x['NUM_STAFFED_BEDS']+x['NUM_ICU_BEDS']+x['PEDI_ICU_BEDS'],axis=1)

beds["BEDS_IN_USE"]=beds.apply(lambda x:x['TOTAL_BEDS']*x["BED_UTILIZATION"],axis=1)

beds["CAN_ADD_MORE"]=beds.apply(lambda x:x["NUM_LICENSED_BEDS"]-x["TOTAL_BEDS"],axis=1)

beds
TBEDS_STATEWISE=pd.DataFrame(beds.groupby('STATE_NAME').BEDS_IN_USE.sum().round(0))

data2=data2.fillna(int("0"))

data2["EXPECTED_PAT"]=data2.apply(lambda x:round(x["Active"]*x["Hospitalization_Rate"]/100,0),axis=1)

BEDS_NEEDED=pd.DataFrame()

BEDS_NEEDED["STATE"]=data2["Province_State"]

BEDS_NEEDED["EXPECTED_PAT"]=data2["EXPECTED_PAT"]

SUMMARY=pd.DataFrame(BEDS_NEEDED.set_index("STATE").join(TBEDS_STATEWISE))

SUMMARY=SUMMARY.fillna(int("0"))

SUMMARY["BEDS_REQ"]=SUMMARY.apply(lambda x: x["EXPECTED_PAT"]-x["BEDS_IN_USE"],axis=1)

print(SUMMARY)
geodata=gpd.read_file("../input/geodata-us-hospital-beds/Definitive_Healthcare_USA_Hospital_Beds.shp")

m1=folium.Map(location=[40.7128,-74.0060],tiles='openstreetmap',zoom_start=10)

#data=data.dropna(subset=["X","Y"],axis=0)

for idx,row in beds.iterrows():

    if row["CAN_ADD_MORE"]>0 and row["HQ_STATE"]=="NY":

        Marker([row.Y,row.X],color="blue",popup=row.HOSPITAL_NAME).add_to(m1)

embed_map(m1,"q-2.html")