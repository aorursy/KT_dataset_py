# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import plotly.graph_objs as go

import cufflinks as cf

cf.set_config_file(offline=True)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data and write first 5 component

data = pd.read_csv("../input/global-terrorism/globalterrorismdb_0718dist.csv",delimiter=',', encoding = "ISO-8859-1")
data.info()
data.head()
#if you want to change columns name, you can use following code

#data.rename(columns={'imonth': 'Month',

# 'iyear': 'Year',

#'eventid': 'Event ID',

#'country_txt': 'Country',

#'region_txt': 'Region',

#'provstate': 'State',

#'city': 'City',

#'attacktype1_txt': 'Attack_type',

# 'targtype1_txt': 'Target',

# 'natlty1_txt': 'Nationality',

#'gname': 'Terrorist Group',

# 'weaptype1_txt': 'Weapon type',

# 'weapsubtype1_txt': 'Weapon subtype',

# 'nkill': 'Killed',

# 'nwound': 'Wounded'},inplace=True)
data.describe()
data.info("columns")
data.country_txt.unique()
data.country_txt.value_counts()

# or you can use data['country_txt'].value_counts()
data.groupby(['country_txt']).agg('sum')
temp= data[(data["country_txt"] == 'Turkey')]

country_list = list(temp)

print(temp)
data.head()
dataframe=data[["iyear","imonth","iday","city","attacktype1_txt","targtype1_txt","targsubtype1","weaptype1","nkill","nwound","suicide","success"]]
f,ax = plt.subplots(figsize=(8,8)) #figure size command

sns.heatmap(dataframe.corr(), annot=False, linewidths=.4, fmt =".1f=", ax=ax) 
f,ax = plt.subplots(figsize=(10,5))

data.nkill.plot(kind="line",color = "b", label="number of kills",linewidth=5, alpha=0.5, grid=True, linestyle=":")

data.nwound.plot(color="g", label="number of wounds",linewidth=5, alpha=0.5, grid=True,linestyle=":")

plt.legend(loc="upper right")

#'best'	0

#'upper right'	1

#'upper left'	2

#'lower left'	3

#'lower right'	4

#'right'	5

#'center left'	6

#'center right'	7

#'lower center'	8

#'upper center'	9

#'center'	10

plt.xlabel("Number of attacks", size=20)

plt.ylabel("Person", size=20)

plt.title("Line Plot")
data.plot(kind="scatter", x="iyear", y="nkill",grid=True, alpha=1, color="r",figsize=(15,5))

plt.xlabel("Year", size= 25)

plt.ylabel("Kills number", size= 25) #size command helps you to change for labels size.

plt.title("Year-Kill Scatter Plot")
#Visualizing the nulls. Each YELLOW cell represents a null

plt.figure(figsize=(10,5))

sns.heatmap(data.isnull(),cmap='viridis',cbar=False)
motive=data['motive']
data=data[['eventid','iyear', 'imonth', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 

           'success', 'attacktype1_txt', 'targtype1_txt', 'natlty1_txt', 'gname', 'weaptype1_txt', 'weapsubtype1_txt', 'nkill', 'nwound']]
data.head()
plt.figure(figsize=(16,6))

sns.heatmap(data.isnull(),cmap='viridis',cbar=False)
#Filling the nulls to save the data

data['weaptype1_txt'].fillna('No Record', inplace=True)

data['natlty1_txt'].fillna('Unknown', inplace=True)
motive.dropna(inplace=True)
print('No. of rows before dropping nulls: {}'.format(data['eventid'].count()))

data.dropna(inplace=True)

print('No. of rows after dropping nulls: {}'.format(data['eventid'].count()))
data['Casualties']=data['nkill']+data['nwound']
data.head()
data = [go.Scattermapbox(

            lat= data['latitude'],

            lon= data['longitude'],

            customdata = data['eventid'],

            mode='markers',

            marker=dict(

                size= 3.5,

                color = 'red',

                opacity = .7,

            ),

          )]

layout = go.Layout(autosize=False,

                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",

                                bearing=0,

                                pitch=50,

                                zoom=2,

                                center= dict(

                                         lat=12,

                                         lon=39),

                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),

                    width=900,

                    height=600, title = "Terrorist attack locations")



fig = dict(data=data, layout=layout)

iplot(fig)