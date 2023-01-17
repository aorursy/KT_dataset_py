# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/AppleStore.csv")

data.head(10)



data = data[["track_name",

             "size_bytes", 

                 "currency", 

                 "price",

                 "rating_count_tot",

                 "rating_count_ver",

                 "user_rating",

                 "user_rating_ver",

                 "cont_rating",

                 "prime_genre",

                "sup_devices.num",

                "lang.num"]]



data
for each in data.columns:

    assert  data[each].notnull().all() # it returns nothing so we don't have missing

                                        # data on dataset.




free_data = data[data.price ==0]

unfree_data = data[data.price > 0]



free_list = []

unfree_list = []

for each in unfree_data["prime_genre"].value_counts().index:

    x = unfree_data[unfree_data["prime_genre"] == each]

    unfree_list.append(sum(x.user_rating)/len(x))







for each in free_data["prime_genre"].value_counts().index:

    x = free_data[free_data["prime_genre"] == each]

    free_list.append(sum(x.user_rating)/len(x))

    

# We found the average of every category and added them to the list

    



f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=free_list,y=data["prime_genre"].value_counts().index,

            color='red',alpha = 0.5,label='free' )

sns.barplot(x=unfree_list,y=data["prime_genre"].value_counts().index,

            color='green',alpha = 0.7,label='nonfree')



ax.set(xlabel='Average using rating', ylabel='application categories',

       title = "Free and Non Free application ")

labels = data["prime_genre"].value_counts().index

fig = {

  "data": [

    {

      "values": data["prime_genre"].value_counts(),

      "labels": labels,

      "domain": {"x": [0, .5]},

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Distribution of Application categories",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Pie Chart",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)


data.size_bytes = data.size_bytes.astype(float)



data.size_bytes = data.size_bytes / 100000000

data.rename(columns = {"size_bytes": "size_gigabytes"}, 

                                 inplace = True)



data.rename(columns = {"sup_devices.num" : "sup_devices_num",

                      "lang.num" : "lang_num"}, inplace = True)



color_list = [each for each in data.lang_num]

data = [

    {

        'y': data.sup_devices_num,

        'x': data.user_rating,

        'mode': 'markers',

        'marker': {

            'color': color_list,

            'size': data.size_gigabytes,

            'showscale': True

        },

        "text" :  data.track_name  

    }

]

iplot(data)
data = pd.read_csv("../input/AppleStore.csv")

filter_price = data.price > 100

new_data = data[filter_price]

new_data



plt.bar(new_data.track_name,new_data.price)

new_data = data[["price", "user_rating"]]



pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=new_data, palette=pal, inner="points")

plt.show()

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()