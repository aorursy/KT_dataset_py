import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import json

import os
with open('/kaggle/input/bikedata/BartStations.txt', 'r') as file:

    bart_json = json.load(file)



bart_json
bart_stations = bart_json['root']['stations']['station']

num_stations = len(bart_stations)

num_stations
name = []

abbr = []

lat = []

lon = []



for i in range(num_stations):

    name.append(bart_stations[i]['name'])

    abbr.append(bart_stations[i]['abbr'])

    lat.append(bart_stations[i]['gtfs_latitude'])

    lon.append(bart_stations[i]['gtfs_longitude'])



bart_df = pd.DataFrame({'name': name, 'abbriviation': abbr, 'latitude': lat, 'longitude': lon})

bart_df.head()
bart_df.to_csv('bart.csv', index = False)
bart_yellow = ['ANTC', 'PCTR', 'PITT', 'NCON', 'CONC', 'PHIL', 

               'WCRK', 'LAFY', 'ORIN', 'ROCK', 'MCAR', '19TH',

               '12TH', 'WOAK', 'EMBR', 'MONT', 'POWL', 'CIVC',

               '16TH', '24TH', 'GLEN', 'BALB', 'DALY', 'COLM',

               'SSAN', 'SBRN', 'SFIA', 'MLBR']

bart_green = ['WARM', 'FRMT', 'UCTY', 'SHAY', 'HAYW', 'BAYF', 'SANL',

              'COLS', 'FTVL', 'LAKE', 'WOAK', 'EMBR', 'MONT', 'POWL', 

              'CIVC', '16TH', '24TH', 'GLEN', 'BALB', 'DALY']

bart_red = ['RICH', 'DELN', 'PLZA', 'NBRK', 'DBRK', 'ASHB', 'MCAR', 

            '19TH', '12TH', 'WOAK', 'EMBR', 'MONT', 'POWL', 'CIVC', 

            '16TH', '24TH', 'GLEN', 'BALB', 'DALY', 'COLM', 'SSAN', 

            'SBRN', 'MLBR']

bart_orange = ['RICH', 'DELN', 'PLZA', 'NBRK', 'DBRK', 'ASHB', 'MCAR',

               '19TH', '12TH', 'LAKE', 'FTVL', 'COLS', 'SANL', 'BAYF',

               'HAYW', 'SHAY', 'UCTY', 'FRMT', 'WARM']

bart_blue = ['DUBL', 'WDUB', 'CAST', 'BAYF', 'SANL', 'COLS', 'FTVL',

             'LAKE', 'WOAK', 'EMBR', 'MONT', 'POWL', 'CIVC', '16TH',

             '24TH', 'GLEN', 'BALB', 'DALY']
def route (rt_list):

    route_df = pd.DataFrame(columns = bart_df.columns)

    for station in rt_list:

        route_df = route_df.append(bart_df[bart_df['abbriviation'] == station])

    return route_df



bart_yellow_df = route(bart_yellow)

bart_yellow_df.head(len(bart_yellow))
bart_green_df = route(bart_green)

bart_green_df.head(len(bart_green))
bart_red_df = route(bart_red)

bart_red_df.head(len(bart_red))
bart_orange_df = route(bart_orange)

bart_orange_df.head(len(bart_orange))
bart_blue_df = route(bart_blue)

bart_blue_df.head(len(bart_blue))
bart_yellow_df.to_csv('bart_yellow.csv', index = False)

bart_green_df.to_csv('bart_green.csv', index = False)

bart_red_df.to_csv('bart_red.csv', index = False)

bart_orange_df.to_csv('bart_orange.csv', index = False)

bart_blue_df.to_csv('bart_blue.csv', index = False)