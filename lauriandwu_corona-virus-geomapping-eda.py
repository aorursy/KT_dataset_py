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
# importing magic functions

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline



# importing visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
# importing the combined data

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

df.head()

df.info()
# dropping columns with redundant data 

df = df.drop(['Last Update','Sno'], axis=1)
# changing the date type to datetime

df['Date'] = df['Date'].astype('datetime64[D]')
# changing data type float to int (there are no half people :-))

df[['Confirmed',"Deaths",'Recovered']] = df[['Confirmed',"Deaths",'Recovered']].astype(int)
# checking for missing values

df.isna().sum()

# fill NA with new string: 'Unknown'

df[['Province/State']] = df[['Province/State']].fillna('Unknown')
# combine China and Mainland China

df['Country'] = df['Country'].replace({'Mainland China':'China'})

# combine Cruise Ship and Diamons Princess Cruise Ship

df['Province/State'] = df['Province/State'].replace({'Cruise Ship':'Diamond Princess cruise ship'})

# replace 'Other' country for the cruise with 'Japan'

df.loc[df['Province/State'] =='Diamond Princess cruise ship', 'Country'] = 'Japan'
# visualize development of Corona Cases over time



f, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(x='Date',y='Confirmed',data=df,ci=None,label="Confirmed", color='B')

sns.lineplot(x='Date',y='Deaths',data=df,label="Deaths", ci=None, color='R')

sns.lineplot(x='Date',y='Recovered',data=df,label="Recovered", ci=None, color='G')



plt.legend(loc="upper left")

plt.xticks(rotation=45)

plt.ylabel('Corona Cases')

plt.xlim('2020-01-22','2020-02-15')

plt.tight_layout()

plt.box(False)

plt.title("Corona Cases over Time", fontweight="bold")

plt.show()
# get the data for the most recent date 

df['Date'].max()

df_now = df[df['Date']=='2020-02-15']



print("As of February 15, 2020 there are",df_now['Confirmed'].sum(),"confirmed Corona cases worldwide.",df_now['Deaths'].sum(),"people have died from the virus and",

      df_now['Recovered'].sum(),"have recovered.")
# Corona Cases by Country

df_now = df_now.groupby('Country', as_index=False).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'}).sort_values(by=['Confirmed'],ascending=False)

df_now = df_now.reset_index(drop=True)

df_now
# loading world coordinates map (found on Kaggle)

df_geo = pd.read_csv("../input/world-coordinates/world_coordinates.csv")
df_geo = df_geo.drop(['Code'], axis=1)
df_geo.head()

df_geo.shape

df_now.head()

df_now.shape
# Merging the 2 dataframes on Country to get the long- and latitude values

df_comb = pd.merge(df_now, df_geo, on='Country',how='left')

df_comb
# Geomapping with Folium

import folium

world_map=folium.Map(location=[10, -20], zoom_start=0.5,tiles='cartodbdark_matter')

for lat, lon, value, name in zip(df_comb['latitude'], df_comb['longitude'], df_comb['Confirmed'], df_comb['Country']):

    folium.CircleMarker([lat, lon],radius=7,popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                                     '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.5 ).add_to(world_map)

world_map