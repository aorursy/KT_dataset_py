import numpy as np

import pandas as pd



import plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
# Access token

from distutils.dir_util import copy_tree

copy_tree(src = "../input/tokens/", dst = "../working")



from access_tokens import *

mapbox_access_token = get_mapbox_token()
dataset = pd.read_csv('../input/chicago-food-inspections/food-inspections.csv')

dataset.head(5)
latest_data = dataset.sort_values('Inspection Date', ascending = False).groupby('License #').head(1)

latest_data.dropna(subset=['Risk', 'Facility Type', 'DBA Name', 'Latitude', 'Longitude'], axis = 0, how = 'all', inplace = True)

latest_data = latest_data[(latest_data['Results'] != 'Out of Business') & (latest_data['Results'] != 'Business Not Located')]

latest_data['Name'] = latest_data.apply(lambda row: row['AKA Name'] if not pd.isnull(row['AKA Name']) else row['DBA Name'], axis = 1)

latest_data['Name'] = latest_data['Name'] + '<br>' + latest_data['Address']
risk_color_map = { "All": "rgb(0, 0, 0)", "Risk 1 (High)": "rgb(255, 0, 0)", "Risk 2 (Medium)": "rgb(204, 204, 0)", "Risk 3 (Low)": "rgb(0, 100, 0)" }

latest_data['Risk Color'] = latest_data['Risk'].map(risk_color_map)



inspection_color_map = { 

    "Pass": "rgb(0, 255, 0)", 

    "Pass w/ Conditions": "rgb(0, 255, 0)",

    "Fail": "rgb(255, 0, 0)", 

    "No Entry": "rgb(255, 0, 0)", 

    "Not Ready": "rgb(255, 0, 0)" }

latest_data['Inspection Color'] = latest_data['Results'].map(inspection_color_map)

    

latest_data.reset_index(inplace=True)

print("Total businesses: {}".format(latest_data.shape[0]))
facility_types = latest_data['Facility Type'].value_counts().keys().tolist()

facility_count = latest_data['Facility Type'].value_counts().tolist()



final_types = []

final_count = []

others_count = 0

one_percent = 0.01 * latest_data.shape[0]

for count, facility_type in zip(facility_count, facility_types):

    if count > one_percent:

        final_types.append(facility_type)

        final_count.append(count)

    else:

        others_count += count

        

final_types.append('Others')

final_count.append(others_count)



# figure

fig = {

    "data": [{

        "values": final_count,

        "labels": final_types,

        "hoverinfo": "label+percent",

        "hole": .5,

        "type": "pie"

        },

    ],

    "layout": {

        "title": "Types of facilities",

        "width": 800,

        "height": 800

    }

}



iplot(fig)
data = [

    go.Scattermapbox(

        lat = latest_data['Latitude'],

        lon = latest_data['Longitude'],

        text = latest_data['Name'],

        hoverinfo = 'text',

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            color = latest_data['Risk Color'],

            opacity = 0.7,

            size = 4

        )

    )

]



layout = go.Layout(

    mapbox = dict(

        accesstoken = mapbox_access_token,

        zoom = 10,

        center = dict(

            lat = 41.8781,

            lon = -87.6298

        ),

    ),

    height = 800,

    width = 800,

    title = "Facilities in Chicago")



fig = go.Figure(data, layout)

iplot(fig, filename = 'facilities')
latest_data['Risk'].value_counts()
data = [

    go.Bar(

        x = latest_data['Results'].value_counts().keys().tolist(),

        y = latest_data['Results'].value_counts().tolist(),

        marker = dict(

            color = [

                'rgb(0,100, 0)', 

                'rgb(0,100, 0)',

                'rgb(255, 0, 0)',

                'rgb(255, 0, 0)',

                'rgb(255, 0, 0)'

            ]

        )

    )

]



layout = go.Layout(

    title = 'Inspection Results',

)



fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = 'inspections')
data = [

    go.Scattermapbox(

        lat = latest_data['Latitude'],

        lon = latest_data['Longitude'],

        text = latest_data['Name'],

        hoverinfo = 'text',

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            color = latest_data['Inspection Color'],

            opacity = 0.7,

            size = 4

        )

    )

]



layout = go.Layout(

    mapbox = dict(

        accesstoken = mapbox_access_token,

        zoom = 10,

        center = dict(

            lat = 41.8781,

            lon = -87.6298

        ),

    ),

    height = 800,

    width = 800,

    title = "Facilities in Chicago")



fig = go.Figure(data, layout)

iplot(fig, filename = 'facilities')
passed_inspections = latest_data[(latest_data['Results'] == 'Pass') | (latest_data['Results'] == 'Pass w/ Conditions')]

failed_inspections = latest_data[(latest_data['Results'] == 'Fail') | (latest_data['Results'] == 'No Entry') | (latest_data['Results'] == 'Not Ready')]



trace0 = go.Bar(

        x = passed_inspections.groupby('Wards').size().keys(),

        y = passed_inspections.groupby('Wards').size().tolist(),

        name = 'Passed inspections',

        marker = dict(

            color = 'rgb(55, 83, 109)'

        )

    )



trace1 = go.Bar(

        x = failed_inspections.groupby('Wards').size().keys(),

        y = failed_inspections.groupby('Wards').size().tolist(),

        name = 'Failed inspections',

        marker = dict(

            color = 'rgb(26, 118, 255)'

        )

    )



data = [trace0, trace1]

layout = go.Layout(

    title = 'Inspection Results',

)



fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = 'ward-wise-inspections')
import re

violators = latest_data.dropna(subset=['Violations'], axis = 0, how = 'all')

violations = violators.apply(lambda row: re.findall('\|\s([0-9]+)[.]', str(row['Violations'])), axis = 1)

first_violations = violators.apply(lambda row: row['Violations'].split('.')[0], axis = 1)



for violation, first_violation in zip(violations, first_violations):

    violation.append(first_violation)



flat_list = [item for sublist in violations for item in sublist]

unique, counts = np.unique(flat_list, return_counts=True)
violation = []

violation_count = []

for value, count in zip(unique, counts):

    if count > 100:

        violation.append(unique)

        violation_count.append(count)
data = [

    go.Bar(

        x = violation,

        y = violation_count,

        marker = dict(

            color = 'rgb(55, 83, 109)'

        )

    )

]



layout = go.Layout(

    title = 'Majority Violations',

)



fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = 'violations')
from math import cos, asin, sqrt

def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295     #Pi/180

    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2

    return 12742 * asin(sqrt(a))
def get_plot(dataset, curr_latitude = 41.8781, curr_longitude = -87.6298, risk_level = 'Low', search_distance = 5):

    dataset = dataset[dataset['Facility Type'] == 'Restaurant']

    

    if (risk_level == 'Low'):

        dataset = dataset[dataset['Risk'] == "Risk 3 (Low)"]

    elif (risk_level == 'Medium'):

        dataset = dataset[(dataset['Risk'] == "Risk 3 (Low)") | (dataset['Risk'] == "Risk 2 (Medium)")]

    elif (risk_level == 'High'):

        dataset = dataset[dataset['Risk'] != "All"]

    

    dataset = dataset[dataset.apply(lambda row: distance(curr_latitude, curr_longitude, row['Latitude'], row['Longitude']) < search_distance, axis = 1)]

    dataset.reset_index(inplace = True)

    

    data = [

        go.Scattermapbox(

            lat = dataset['Latitude'],

            lon = dataset['Longitude'],

            text = dataset['Name'],

            hoverinfo = 'text',

            mode = 'markers',

            marker = go.scattermapbox.Marker(

                color = dataset['Risk Color'],

                opacity = 0.7,

                size = 4

            )

        )

    ]



    layout = go.Layout(

        mapbox = dict(

            accesstoken = mapbox_access_token,

            zoom = 10,

            center = dict(

                lat = curr_latitude,

                lon = curr_longitude

            ),

        ),

        height = 800,

        width = 800,

        title = "Searched Restaurants in Chicago based on location and distance")



    fig = go.Figure(data, layout)

    iplot(fig, filename='restaurants')
get_plot(latest_data, 41.8781, -87.6298, 'Medium', 5)
# Removing token

from IPython.display import clear_output

clear_output(wait=True)

!rm -rf ../working/access_tokens.py