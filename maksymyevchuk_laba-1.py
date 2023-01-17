# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
rest_country = pd.read_excel("../input/Country-Code.xlsx")





rest_country.columns=['country code','country']



print (rest_country.head())





rest_data=pd.read_csv('../input/zomato.csv',encoding='latin-1') #пришлось менять кодировку из-за того  то была ошибка чтения utf-8



rest_data.columns=[x.lower() for x in rest_data.columns]



print (rest_data.head())
rest_all_data = pd.merge(rest_data,rest_country,on='country code',how='inner')

pd.merge(rest_data,rest_country,on='country code',how='inner')






# rest_all_data.shape

# rest_all_data['country'].unique()

# print(rest_all_data['country'].value_counts())





# rest_all_data['country'].value_counts().plot(kind='bar',title='Total Restaurants On Zomato In Countries'

#                                              ,figsize=(20,10),fontsize=20)



labels = list(rest_all_data['country'].value_counts().index)

values = list(rest_all_data['country'].value_counts().values)



fig = {

    "data":[

        {

            "labels" : labels,

            "values" : values,

            "hoverinfo" : 'label+percent',

            "domain": {"x": [0, .9]},

            "hole" : 0.6,

            "type" : "pie",

            "rotation":120,

        },

    ],

    "layout": {

        "title" : "Zomato's Presence around the World",

        "annotations": [

            {

                "font": {"size":20},

                "showarrow": True,

                "text": "Countries",

                "x":0.2,

                "y":0.9,

            },

        ]

    }

}



iplot(fig)
# print(rest_all_data['city'].value_counts())

# rest_all_data['city'].value_counts().plot(kind='bar',title='Total Restaurants On Zomato In Cities',figsize=(20,10),fontsize=20)

labels = list(rest_all_data['city'].value_counts().index)

values = list(rest_all_data['city'].value_counts().values)



fig = {

    "data":[

        {

            "labels" : labels,

            "values" : values,

            "hoverinfo" : 'label+percent',

            "domain": {"x": [0, .9]},

            "hole" : 0.6,

            "type" : "pie",

            "rotation":50,

        },

    ],

    "layout": {

        "title" : "Zomato's Presence around the World(city)",

        "annotations": [

            {

                "font": {"size":20},

                "showarrow": True,

                "text": "Города",

                "x":0.2,

                "y":0.9,

            },

        ]

    }

}



iplot(fig)
# sns.countplot(x = "aggregate rating", hue = "cuisines", data = rest_all_data) В данной среде перегружает процессор из-за большого количесва данных
rest_all_data['cuisines'].unique()
Cuisine_data =rest_all_data.groupby(['cuisines'], as_index=False)['restaurant id'].count()

Cuisine_data.columns = ['cuisines', 'Number of Resturants']

Top15= (Cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(15)

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.barplot(Top15['cuisines'], Top15['Number of Resturants'])

plt.xlabel('cuisines', fontsize=20)

plt.ylabel('Number of Resturants', fontsize=20)

plt.title('Top 15 cuisines on Zomato', fontsize=30)

plt.xticks(rotation = 90)

plt.show()





correlation = rest_all_data.groupby('restaurant name',as_index=False)[['aggregate rating','price range']].mean().round(2).sort_values(ascending=False,by='aggregate rating')

weight = correlation['aggregate rating']

height = correlation['price range']

plt.figure(figsize=(10,8))

plt.scatter(weight,height,c='g',marker='o')

plt.xlabel('Average Rating')

plt.ylabel('Price range')

plt.title('Average Rating Vs Price range')

plt.show()
sns.countplot(x = "aggregate rating", hue = "has table booking", data = rest_all_data)
#Исходя из описания • Aggregate Rating: Average rating out of 5

# • Rating color: depending upon the average rating color

# • Rating text: text on the basis of rating of rating

# Зависимость указанных в задании параметров дает возможность анализировать их используя рейтинг.

# При этом мы не теряем гамму цветов для 'rating color'



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.graph_objs as go





plot_data = [dict(

    type='scattergeo',

    lon = rest_all_data['longitude'],

    lat = rest_all_data['latitude'],

    text = rest_all_data['restaurant name'],

    mode = 'markers',

    marker = dict(

   cmin = 0,

    color = rest_all_data['aggregate rating'],

    cmax = rest_all_data['aggregate rating'].max(),

    colorbar=dict(

                title="Rating"

            )

    )

    

)]

layout = dict(

    title = 'Dependence on rating color',

    hovermode='closest',

    geo = dict(showframe=False, countrywidth=1, showcountries=True,

               showcoastlines=True, projection=dict(type='mercator'))

)

fig = go.Figure(data=plot_data, layout=layout)

iplot(fig)
grouped_a=rest_all_data[rest_all_data['country code']==1].groupby('locality')

locality_dict={}



for key,item in grouped_a:



    total_votes=item['votes'].sum()

    w=(item['votes']*item['aggregate rating']).sum()

    ans=w/total_votes

    locality_dict[key]=ans.round(3)

top_values=sorted(locality_dict,key=locality_dict.get,reverse=True)[0:10]

print("----------------------------------------------------")

print("  Weighted User Rating for Localities in India")

print("----------------------------------------------------")

for i in top_values:

    print("| {:37s} | {:3f} |".format(i,locality_dict[i]))

    print("----------------------------------------------------")

# Gouping data of All localities

grouped=rest_all_data.groupby('locality')

locality_dict={}

print()

for key,item in grouped:

    total_votes=item['votes'].sum()

    w=(item['votes']*item['aggregate rating']).sum()

    ans=w/total_votes

    locality_dict[key]=ans.round(3)

top_values=sorted(locality_dict,key=locality_dict.get,reverse=True)[0:10]

print("---------------------------------------------------------")

print("  Weighted User Rating for Localities All over the world")

print("---------------------------------------------------------")

for i in top_values:

    print("| {:42s} | {:3f} |".format(i,locality_dict[i]))

    print("---------------------------------------------------------")
# Handling Duplicate values of Restaurant Names

rest_all_data['restaurant name']=rest_all_data['restaurant name'].replace("Giani's","Giani")

# Dropping NaN values

rest_all_data['restaurant name'].dropna(inplace=True)

restaurant_names=rest_all_data['restaurant name']

r_dict={}

# Creating the no. of outlets dictionary

for i in restaurant_names:

    if i in r_dict:

        r_dict[i]+=1

    else:

        r_dict[i]=1

plt.figure(figsize=(10,8))

# Sorting the values on the basis of no of outlets

outlets_sorted=sorted(r_dict,key=r_dict.get,reverse=True)[0:15]

print("-----------------------------------------------")

print("  Restaurants Having Highest Number Of Outlets")

print("-----------------------------------------------")

print("| {:20s} | {:14s} |".format('Restaurant Name','No. Of Outlets'))

print("-----------------------------------------")

for i in outlets_sorted:

    print("| {:20s} | {:10d}     |".format(i,r_dict[i]))

    print("-----------------------------------------")

    # Plotting the graph

    plt.barh(width=r_dict[i],y=i,height=0.7)

    plt.text(y=i,x=r_dict[i]+1,s=r_dict[i])

plt.title('Top 15 Restaurants having highest no. of outlets')



plt.xlabel('No Of Outlets')

plt.ylabel("Restaurant Names")

plt.show()
