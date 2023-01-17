import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings



import folium 

import folium.plugins as plugins

import branca



warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
indiv=pd.read_csv("../input/covid19india/IndividualDetails.csv")

indiv.head(3)
col=indiv.columns.tolist()

idf=indiv.drop(['id','unique_id','government_id','diagnosed_date','contacts'],axis=1)

idf.head(2)
idf.isna().sum()
idf=idf.fillna(0)

idf.isna().sum()
idf.head()
plt.figure(figsize=(40,5))

sns.countplot(idf['detected_state'])
st_name=np.unique(idf['detected_state'])

print(st_name)

st=input("Enter State: ")

idf_st=idf[idf['detected_state']==st]

plt.figure(1)

sns.countplot(idf_st['current_status'])

plt.figure(2)

sns.countplot(idf_st['gender'])
plt.figure(figsize=(20,5))

sns.countplot(idf_st['detected_district'])
st_name=np.unique(idf['detected_state'])

print(st_name)

for i in st_name:

    print("Current State: ",i)

    idf_st=idf[idf['detected_state']==i]

    plt.figure()

    plt.title("Current Status for {}".format(i))

    sns.countplot(idf_st['current_status'])

    plt.figure()

    plt.title("Status for effected by Genger for {}".format(i))

    sns.countplot(idf_st['gender'])

    plt.figure(figsize=(20,5))

    plt.title("Current Status for {}(Districtwise)".format(i))

    sns.countplot(idf_st['detected_district'])
state=input("Enter State Name: ")

print("Migration Mamp: for {}".format(state))

idf_st=idf[idf['detected_state']==state]

citypt=idf_st['detected_city_pt'].tolist()

mapcity=[]

for i in citypt:

    mapcity.append(i[16:])

print(mapcity)
lat=[]

long=[]



for i in range(len(mapcity)):

    mapcity[i]=mapcity[i].split()

locar=np.array(mapcity).flatten()

for i in range(len(locar)):

    l=''

    if i%2==0:

        for j in locar[i]:

            if j=='(' or j==")":

                continue

            else:

                l+=j

        lat.append(eval(l))

    else:

        for j in locar[i]:

            if j=='(' or j==")":

                continue

            else:

                l+=j

        long.append(eval(l))

    

print("Lattitudes: \n",lat)

print("\nLongitudes: \n",long)
print("Map for perple migrated/returned to {}".format(state))

locations=[lat[0],long[0]]

m = folium.Map(location=locations,zoom_start=12)

for i in range(len(lat)):

    show="Detected Loc=> Lat:"+str(lat[i])+" long:"+str(long[i])

    popup = folium.Popup(show, parse_html=True) 

    folium.Marker([lat[i],long[i]],popup=popup,

                  icon=folium.Icon(color='blue')

                 ).add_to(m)

m
plt.title("Central Tendency of Age groups for {}".format(state))

sns.distplot(idf_st['age'])