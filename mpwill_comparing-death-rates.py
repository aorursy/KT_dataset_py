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

import pandas as pd

import numpy as np

import matplotlib.pylab as pylab

%matplotlib inline



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

import datetime
country_df = pd.read_csv("../input/country-data/country_pop_data1.csv")

adj_cases = pd.read_csv("../input/adjusted-case-nos/adjusted_case_nos.csv")

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 16,

        }
italy = country_df[country_df["Country/Region"]=="Italy"]

spain = country_df[country_df["Country/Region"]=="Spain"]

hubei = country_df[country_df["Province/State"]=="Hubei"]

china = country_df[country_df["Country/Region"]=="China"]

not_hubei= china[china["Province/State"]!="Hubei"]

not_hh = not_hubei[not_hubei["Province/State"]!= "Hong Kong"]

hongkong = country_df[country_df["Province/State"]=="Hong Kong"]

japan = country_df[country_df["Country/Region"]=="Japan"]

s_korea = country_df[country_df["Country/Region"]=="Korea, South"]

sing = country_df[country_df["Country/Region"]=="Singapore"]

iran = country_df[country_df["Country/Region"]=="Iran"]

us = country_df[country_df["Country/Region"]=="US"]

uk = country_df[country_df["Country/Region"]=="United Kingdom"]

ger = country_df[country_df["Country/Region"]=="Germany"]
# Death Rate in Hubei

name = hubei

fig = plt.figure(figsize=(25,5))

plt.ylabel('Death Rate')

plt.xlabel('Date')

plt.title("Death Rates in Hubei, China",fontdict=font)

plt.scatter(name["date"],name["Death Rate"],s=100,marker =8, c='tab:blue',label = 'Death Rate')

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Death Rates in Hubei, China" )
# Active cases in Hubei

name = hubei

fig = plt.figure(figsize=(25,5))

plt.ylabel('Active Cases')

plt.xlabel('Date')

plt.title("Active Cases in Hubei, China",fontdict=font)

plt.scatter(name["date"],name["Active cases"],s=100,marker =8, c='tab:orange',label = 'Hubei')



fig.autofmt_xdate(rotation=75)

plt.legend()

print("Active Cases in Hubei, China" )
#Death rates for provinces in China except for Hubei

name = not_hubei

fig = plt.figure(figsize=(25,5))

plt.ylabel('Death Rate')

plt.xlabel('Date')

plt.title("Death Rates in Chinese provinces except for Hubei",fontdict=font)

plt.scatter(name["date"],name["Death Rate"],s=100,marker =8, c='tab:blue',label = 'Death Rate')

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Death Rates in Chinese provinces except for Hubei")
# Active Cases in Chinese provinces except for Hubei

name = not_hh

fig = plt.figure(figsize=(25,5))

plt.ylabel('Active Cases')

plt.xlabel('Date')

plt.title("Active Cases in Chinese provinces except for Hubei",fontdict=font)

plt.scatter(hongkong["date"],hongkong["Active cases"],s=100,marker =8, c='tab:green',label = 'Active cases in Hong Kong')

plt.scatter(name["date"],name["Active cases"],s=100,marker =8, c='tab:orange',label = 'Active Cases (other provinces)')

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Active Cases in Chinese provinces except for Hubei")
#death rates south korea

name = s_korea

fig = plt.figure(figsize=(25,5))

plt.ylabel('Death Rate')

plt.xlabel('Date')

plt.title("Death Rates in South Korea",fontdict=font)

plt.scatter(name["date"],name["Death Rate"],s=100,marker =8, c='tab:blue',label = 'Death Rate')

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Death Rates in South Korea")
#death rates Italy, Spain, Germany



fig = plt.figure(figsize=(25,5))

plt.ylabel('Death Rate')

plt.xlabel('Date')

plt.title("Death Rates in Italy, Spain and Germany",fontdict=font)

plt.scatter(italy["date"],italy["Death Rate"],s=100,marker =8, c='tab:red',label = 'Italy')



plt.scatter(spain["date"],spain["Death Rate"],s=100,marker =8, c='tab:green',label = 'Spain')

plt.scatter(ger["date"],ger["Death Rate"],s=100,marker =8, c='tab:cyan',label = 'Germany')

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.xlim((20,120))

print("Death Rates in Italy, Spain and Germany")
# Active Cases in Italy, Spain & Germany



fig = plt.figure(figsize=(25,5))

plt.ylabel('Active Cases')

plt.xlabel('Date')

plt.title("Active Cases in Italy, Spain & Germany",fontdict=font)

plt.scatter(italy["date"],italy["Active cases"],s=100,marker =8, c='tab:red',label = 'Italy')

plt.scatter(spain["date"],spain["Active cases"],s=100,marker =8, c='tab:green',label = 'Spain')

plt.scatter(ger["date"],ger["Active cases"],s=100,marker =8, c='tab:cyan',label = 'Germany')

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.xlim((20,120))

print("Active Cases in Italy, Spain and Germany")
adj_cases1 = adj_cases.loc[adj_cases["Death Rate"]>=6]

#adj_cases1.head(3)

adj_cases1.drop(['Unnamed: 0'],axis=1,inplace=True)
# Counties that are doubling on a world map

import folium

import math



#doubling = doubling[doubling['Doubling Rate'] == doubling['Doubling Rate'].max()]

map = folium.Map(location=[10, 0], tiles = "cartodbpositron", zoom_start=3.0,max_zoom=6,min_zoom=2)

for i in range(0,len(adj_cases1)):

     

    folium.Circle(location=[adj_cases1.iloc[i,3],

                            adj_cases1.iloc[i,4]],

                           

                            tooltip = "<h5 style='text-align:center;font-weight: bold'>"+adj_cases1.iloc[i]['Country/Region']+"</h5>"+

                            "<div style='text-align:center;'>"+str(np.nan_to_num(adj_cases1.iloc[i]['State/Province']))+"</div>"+

                            "<hr style='margin:10px;'>"+

                            "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

                            "<li>Reported Cases: "+str(adj_cases1.iloc[i,7])+"</li>"+

                            "<li>as at: "+str(adj_cases1.iloc[i,6])+"</li>"+

                            "<li>Death Rate: "+str(adj_cases1.iloc[i,10])+"</li>"+

                            "<li>Minimum Adjusted Cases: "+str(int(adj_cases1.iloc[i,11]))+"</li>"+

                            "<li>Max Adjusted Cases: "+str(int(adj_cases1.iloc[i,12]))+"</li>"+

                            "</ul>"

                            ,

                            #radius=(math.sqrt(adj_cases.iloc[i,5])*4000 ),

                            radius=(int((np.log(adj_cases1.iloc[i,5]+1.00001)))+0.2)*10000,

                            color='orange',

                            fill=True,

                            fill_color='orange').add_to(map)

print("Adjusted case numbers based on death rates")

map    
