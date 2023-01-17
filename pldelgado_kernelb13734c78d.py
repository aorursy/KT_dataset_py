import pandas as pd

import numpy as np

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import plotly.graph_objs

init_notebook_mode()



from plotly.grid_objs import Grid, Column

from plotly.tools import FigureFactory as FF 





import time

from datetime import datetime





# Import dataset

rawdataset = pd.read_csv("../input/DailyTreasuryYieldCurveRateDataAllV3.csv", sep=";", decimal=",", encoding="UTF-8")

# datetime format



dataset = rawdataset[["NEW_DATE", "BC_3MONTH", "BC_6MONTH", 

                      "BC_1YEAR", "BC_2YEAR", "BC_3YEAR", "BC_5YEAR", "BC_7YEAR", 

                      "BC_10YEAR", "BC_20YEAR", "BC_30YEAR"]]

dataset["NEW_DATE"] = dataset["NEW_DATE"].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M'))



# add day column



dataset['day'] =np.arange(len(dataset)) 



# rename columns



dataset.columns = ["date", "3_month", "6_month",

                   "1_year", "2_year", "3_year", "5_year", "7_year", "10_year", "20_year", "30_year",  "day"]



# split dataset



dataset = dataset[["date", "3_month", "6_month", 

                   "1_year", "2_year", "3_year", "5_year", "7_year", "10_year", "20_year", "30_year",  "day"]]



# create new dataset filttering per bond´s duration



dataset_3_month = dataset [["date", "3_month", "day"]]

dataset_3_month['term'] = '3_month'

dataset_3_month['duration'] = 3

dataset_3_month.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_6_month = dataset [["date", "6_month", "day"]]

dataset_6_month['term'] = '6_month'

dataset_6_month['duration'] = 4

dataset_6_month.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_1_year = dataset [["date", "1_year", "day"]]

dataset_1_year['term'] = '1_year'

dataset_1_year['duration'] = 5

dataset_1_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_2_year = dataset [["date", "2_year", "day"]]

dataset_2_year['term'] = '2_year'

dataset_2_year['duration'] = 6

dataset_2_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_3_year = dataset [["date", "3_year", "day"]]

dataset_3_year['term'] = '3_year'

dataset_3_year['duration'] = 7

dataset_3_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_5_year = dataset [["date", "5_year", "day"]]

dataset_5_year['term'] = '5_year'

dataset_5_year['duration'] = 8

dataset_5_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_7_year = dataset [["date", "7_year", "day"]]

dataset_7_year['term'] = '7_year'

dataset_7_year['duration'] = 9

dataset_7_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_10_year = dataset [["date", "10_year", "day"]]

dataset_10_year['term'] = '10_year'

dataset_10_year['duration'] = 10

dataset_10_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_20_year = dataset [["date", "20_year", "day"]]

dataset_20_year['term'] = '20_year'

dataset_20_year['duration'] = 11

dataset_20_year.columns = ["date", 'rate', 'day', 'term', 'duration']



dataset_30_year = dataset [["date", "30_year", "day"]]

dataset_30_year['term'] = '30_year'

dataset_30_year['duration'] = 12

dataset_30_year.columns = ["date", 'rate', 'day', 'term', 'duration']



# create dataset append



dataset = dataset_3_month.append(dataset_6_month)

dataset = dataset.append(dataset_1_year)

dataset = dataset.append(dataset_2_year)

dataset = dataset.append(dataset_3_year)

dataset = dataset.append(dataset_5_year)

dataset = dataset.append(dataset_7_year)

dataset = dataset.append(dataset_10_year)

dataset = dataset.append(dataset_20_year)

dataset = dataset.append(dataset_30_year, ignore_index=True)



# add pop column. 



dataset['pop'] = 50000000



# values to continent column



dataset['continent'] = 'USA'



# rename dataset columns



dataset.columns=["year", 'gdpPercap', 'day', 'country', 'lifeExp', 'pop', 'continent']



# datetime format column



dataset['year'] = dataset['year'].dt.strftime("%Y")+dataset['year'].dt.strftime("%m")+dataset['year'].dt.strftime("%d")



# str year to int year to run the following codes. must be int to run correctly



dataset['year'] = dataset['year'].apply(int) 



# category country = bonds term



dataset["country"] = dataset["country"].astype("category") 

dataset["country"] = dataset["country"].cat.reorder_categories(["3_month", "6_month",

                   "1_year", "2_year", "3_year", "5_year", "7_year", "10_year", "20_year", "30_year"])



# values to lifeExp column



dataset["lifeExp"] = dataset["country"]



# sort dataset by year and country



dataset = dataset.sort_values(by=["year", "country"])



# take a short piece of dataset. Bigger dataset spend lot of time running (cause of the loop for?) . Code must be improved. Lets go!



dataset = dataset[:][70000:]
# create grid to start visualization



years_from_col = set(dataset['year'])

years_ints = sorted(list(years_from_col))

years = [str(year) for year in years_ints]



# make list of continents

continents = []

for continent in dataset['continent']:

    if continent not in continents: 

        continents.append(continent)



df = pd.DataFrame()



# make grid

for year in years:

    for continent in continents:

        dataset_by_year = dataset[dataset['year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]

        for col_name in dataset_by_year_and_cont:

            # each column name is unique

            temp = '{year}+{continent}+{header}_grid'.format(

                year=year, continent=continent, header=col_name

            )

            #if dataset_by_year_and_cont[col_name].size != 0:

            df = df.append({'value': list(dataset_by_year_and_cont[col_name]), 'key': temp}, ignore_index=True)

# create figure object to make the plot



figure = {  # plot format

    'data': [],

    'layout': {},

    'frames': [],

    

}



# fill in most of layout



figure['layout']['xaxis'] = {'title': 'Bonds Term', 'gridcolor': '#FFFFFF'} 

figure['layout']['yaxis'] = {"range" : [0, 5], 'title': 'Rate', 'gridcolor': '#FFFFFF'} 

figure['layout']['hovermode'] = "closest" # houver mode

figure['layout']['plot_bgcolor'] = 'rgb(223, 232, 243)'







# slides input



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Date:', 

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 100, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



figure['layout']['sliders'] = [sliders_dict] 







# buttons layout, speed, format



figure['layout']['updatemenus'] = [

    {

        'buttons': [ 

            {

                'args': [None, {'frame': {'duration': 100, 'redraw': False}, # speed

                         'fromcurrent': True, # if True, plot start form where is paused

                'transition': {'duration': 0, 'easing': 'quadratic-in-out'}}], 

                'label': 'Play', 

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 

                        'mode': 'immediate',

                'transition': {'duration': 0}}], # no tiene duracion la pausa lo cual tiene sentido en el caso de pause

                'label': 'Pause', # nombre del boton: pause

                'method': 'animate'

            }

        ],

        'direction': 'left', # buttons layout

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



custom_colors = { # plot colors

    'USA': 'rgb(171, 99, 250)',

    

}





# set first year to start loop for



year = dataset["year"].values[0]





# create each data_dict to plot



col_name_template = '{year}+{continent}+{header}_grid'

for continent in continents:

    data_dict = {

        'x': df.loc[df['key']==col_name_template.format(

            year=year, continent=continent, header='lifeExp'

        ), 'value'].values[0],

        'y': df.loc[df['key']==col_name_template.format(

            year=year, continent=continent, header='gdpPercap'

        ), 'value'].values[0],

        'mode': 'markers',

        'text': df.loc[df['key']==col_name_template.format(

            year=year, continent=continent, header='country'

        ), 'value'].values[0],

        'marker': {

            'sizemode': 'area',

            'sizeref': 200000,

            'size': df.loc[df['key']==col_name_template.format(

                year=year, continent=continent, header='pop'

            ), 'value'].values[0],

            'color': custom_colors[continent]

        },

        'name': continent

    }

    figure['data'].append(data_dict)

    



    

# create each data_dict to plot to finish



for year in years:

    frame = {'data': [], 'name': str(year)}

    for continent in continents:

        data_dict = {

            'x': df.loc[df['key']==col_name_template.format(

                year=year, continent=continent, header='lifeExp'

            ), 'value'].values[0],

            'y': df.loc[df['key']==col_name_template.format(

                year=year, continent=continent, header='gdpPercap'

            ), 'value'].values[0],

            'mode': 'markers',

            'text': df.loc[df['key']==col_name_template.format(

                year=year, continent=continent, header='country'

            ), 'value'].values[0],

            'marker': {

                'sizemode': 'area',

                'sizeref': 200000,

                'size': df.loc[df['key']==col_name_template.format(

                    year=year, continent=continent, header='pop'

                ), 'value'].values[0],

                'color': custom_colors[continent]

            },

            'name': continent

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame) #this block was indented and should not have been.

    

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 10, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 0}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)



figure['layout']['sliders'] = [sliders_dict]



config = {'scrollZoom': True}

iplot(figure, config=config) # final plot