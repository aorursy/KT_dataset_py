# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from mpl_toolkits.basemap import Basemap

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





        # Any results you write to the current directory are saved as output.
#Examining a sample ..



pizza_df = pd.read_csv("../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv")

pizza_df.head()
#Check for nulls in main fields 

pizza_df.isnull().sum()
#No.of unique cities ..

pizza_df['city'].nunique()
pizza_df['city'].value_counts().head(1)
pizza_df.head()
pizza_df['menus.name'].unique()
#Rare Pizza(with occurence = 1)

city_pizza=pizza_df[['id','city','latitude','longitude','name','menus.name']]

s1=city_pizza['menus.name'].value_counts()

s2=s1[s1 == 1].index.tolist()

subs="Pizza,"

s3=[x for x in s2 if not re.search(subs, x)]

#s4=np.random.choice(s3,20)

plt_data=city_pizza[city_pizza['menus.name'].isin(s3)]
#plt.scatter(x=plt_data['longitude'],y=plt_data['latitude'])

plt_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)

plt.show()
fig = go.Figure(data=go.Scattergeo(

        lon = pizza_df['longitude'],

        lat = pizza_df['latitude'],

        text = pizza_df['city'],

        mode = 'markers',

        ))

fig.update_layout(

        title = 'Cities with outlets serving unique Pizzas<br>(Hover for city names)',

        geo_scope='usa',

    )

fig.show()