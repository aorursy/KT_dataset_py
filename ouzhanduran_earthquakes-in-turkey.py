#Let's start with adding libraries I will use



import numpy as np 

import pandas as pd 

import seaborn as sns



# plotly

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt



from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Upload Data

earthquake = pd.read_csv("../input/earthquake.csv")

earthquake.head()
earthquake.info()
Richter_mean = earthquake.Richter.mean()

earthquake["Severity_level"] = ["Heavy" if Richter_mean < each else "Harmless" for each in earthquake.Richter]



xm_mean = earthquake.xm.mean()

earthquake["Xm_level"] = ["High" if xm_mean < each else "Low" for each in earthquake.xm]



Depth_mean = 60

earthquake["Depth_level"] = ["Deep" if Depth_mean < each else "Shallow" for each in earthquake.Depth]



print(Richter_mean)

print(xm_mean)

print(Depth_mean)



earthquake.head()
# I  want to see how many different Direction we have.

print(earthquake.Direction.unique())
#Heatmap

data1 = earthquake[["Depth","xm","md","Richter","mw","ms","mb"]]



data1.corr()

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt=".1f",ax=ax)

plt.show()
earthquake.boxplot(column="Richter",by="Depth_level")

plt.show()
earthquake.boxplot(column="mw",by="Depth_level")

plt.show()
# I added new column as Violence and it will show violence level.

violence = 5 

earthquake["Violence"] = ["Strong" if violence < each else "Soft" for each in earthquake.mw]

earthquake.head()
print(earthquake["Violence"].value_counts(dropna =False))
earthquake['Date'] =pd.to_datetime(earthquake['Date'])
# I will continue with Strong Earthquake (mw>5), so I filtered as Strong_earthquake data.

strong_earthquake = earthquake[(earthquake.Violence=="Strong")]

strong_earthquake.head()

# MW vs Richter of each state

# visualize

f,ax1 = plt.subplots(figsize =(35,10))

sns.pointplot(x='City',y='mw',data=earthquake,color='lime',alpha=0.8)

sns.pointplot(x='City',y='Richter',data=earthquake,color='red',alpha=0.8)

plt.text(40,0.6,'MW',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'Richter',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('City',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('MW  VS  Richter',fontsize = 20,color='blue')

plt.grid()
g = sns.jointplot(earthquake.mw, earthquake.Richter, kind="kde", size=7)



plt.show()
city = earthquake.City.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)



earthquake.head()
years = [str(each) for each in list(earthquake.Year.unique())]  # str unique years

types = ['Soft', 'Strong']

custom_colors = {

    'Soft': 'rgb(34, 139, 34)',

    'Strong': 'rgb(167, 34, 0)'

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1910',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}

figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



# make data

year = 1910

for ty in types:

    dataset_by_year = earthquake[earthquake['Year'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Violence'] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = earthquake['Long'],

    lat = earthquake['Lat'],

    hoverinfo = 'text',

    text = ty,

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = custom_colors[ty],

        opacity = 0.7),

)

    figure['data'].append(data_dict)

    

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for ty in types:

        dataset_by_year = earthquake[earthquake['Year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Violence'] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_year_and_cont['Long'],

                lat = dataset_by_year_and_cont['Lat'],

                hoverinfo = 'text',

                text = ty,

                mode = 'markers',

                marker=dict(

                    sizemode = 'area',

                    sizeref = 1,

                    size= 10 ,

                    line = dict(width=1,color = "white"),

                    color = custom_colors[ty],

                    opacity = 0.7),

                name = ty

            )

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Earthquake"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)
