# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

dataset.room_type.unique()

dataset.shape

dataset.columns
dataset.head()
dataset.neighbourhood_group.unique()
m=folium.Map(

    location=[40.723509, -73.972471],

    tiles='Stamen Toner',

    zoom_start=12, 

    width='80%',

    height='80%',

 )



def marker_color(neighbourhood_group):

    

    color_picker={'Brooklyn':'#654F6F',

             'Manhattan':'#FF5842',

             'Queens':'#08B2E3',

             'Staten Island':'#336699',

             'Bronx':'#FF5842'

                 }

    color=color_picker[neighbourhood_group]

    return color



for pin in dataset.index[:500]:

    folium.CircleMarker(

        location=[dataset.latitude[pin], dataset.longitude[pin]],

        radius=6,

        color=marker_color(dataset.neighbourhood_group[pin]),

        fill=True,

        fill_color=marker_color(dataset.neighbourhood_group[pin]),

    ).add_to(m)



m
plt.figure(figsize=(25, 5))

plt.subplot(1,3,1)

sns.violinplot(x='room_type', y='price', data=dataset[(dataset.price<500)&(dataset.neighbourhood_group=='Manhattan')])



plt.subplot(1,3,2)

sns.violinplot(x='room_type', y='price', data=dataset[(dataset.price<500)&(dataset.neighbourhood_group=='Brooklyn')])



plt.subplot(1,3,3)

sns.violinplot(x='room_type', y='price', data=dataset[(dataset.price<500)&(dataset.neighbourhood_group=='Queens')])

plt.show()