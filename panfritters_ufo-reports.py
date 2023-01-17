# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(165855)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import folium

import plotly.graph_objects as go

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ufo-sightings/scrubbed.csv')
df.iloc[56566].datetime
df.describe()
df.hist(figsize=(15,15), bins=100)
df.head()
# Only look at a statistically representative sample of 1%

X_train, X_test, y_train, y_test = train_test_split(df, df.datetime, test_size=.99)
m = folium.Map(location=[37.7929552,-122.4678707],

                    zoom_start = 11)

# folium.Marker([37.7929552,-122.4678707],popup='<strong>Starting Location</strong>', tooltop='clicl').add_to(m)

for index, row in X_train.iterrows():

    if row.latitude and row[10]:

        try:

            lat = float(row.latitude)

            lon = float(row[10])

            try:

                time = "<strong>"+row[0]+"</strong>"

                

            except:

                print("Couldn't find time.")

    

            try:

                city = row[1] if row[1] else "not listed"

                state = row[2] if row[2] else "not listed"

                country = row[3] if row[3] else "not listed"

                shape = row[4] if row[4] else "not listed"

                duration = row[5] if row[5] else "not listed"

                duration_txt = row[6] if row[6] else "not listed"

                comments = row[7] if row[7] else "not listed"

                

                folium.Marker([lat, lon], popup="<strong>"+time+"</strong> "+comments).add_to(m) 

            except:

                print("Coudn't resolve something. Avast!")

                folium.Marker([lat, lon], popup=time).add_to(m)    

                

        except:

            print("Couldn't convert to float: lat {} lon {}".format(row.latitude, row[10]))



m
def clean(df):

    lists = []

    for index, row in df.iterrows():

        try:

            lat = float(row.latitude)

            lon = float(row[10])

            try:

                # 10/10/1949 20:30

                

#                 time = datetime.strptime(row[0], "%-m/%-d/%Y %H:%M")

                time = int(row[0].split('/')[2].split(" ")[0])

                lists.append([lat, lon, time])

            except:

                print("Couldn't find time.-{}-".format(row[0]))

        except:

            print("Couldn't convert to float: lat {} lon {}".format(row.latitude, row[10]))

    return pd.DataFrame(lists, columns=['lat', 'lon', 'time'])
# Look at global heatmap where most sightings are reported along with their year

clean_df = clean(df)

fig = go.Figure(go.Densitymapbox(lat=clean_df.lat, lon=clean_df.lon, z=clean_df.time, radius=10))

fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()