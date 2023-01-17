# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Shelf is an altitude where a drone freezes to perform measurements.

# Each shelf is set with start and end row indexes.

shelves_ranges = {'DATA0.TXT': [[(37, 65)], [(87, 117)], [(132, 158)], [(168, 189)],

                  [(198, 221)], [(229, 252)], [(260, 277)], [(288, 304)],

                  [(311, 337)], [(343, 360)], [(368, 394)]],

                  'DATA1.TXT': [[(211, 235), (688, 711)], [(259, 296), (727, 772)], 

                                [(309, 338), (791, 836)], [(852, 901)], [(926, 971)], [(985, 1033)]],

                  'DATA2.TXT': [ [(143, 167), (619, 642)], [(186, 231), (662, 706)], 

                                [(256, 300), (722, 767)], [(321, 367), (781, 826)], 

                                [(381, 428), (839, 885)], [(441, 494), (900, 940)] ],

                  'DATA5.TXT': [ [(123, 150)], [(164, 189)], [(200, 223)], [(235, 258)], [(273, 298)], [(310, 335)] ],

                  'DATA6.TXT': [ [(101, 124)], [(136, 184)], [(197, 243)], [(253, 303)], [(313, 366)], [(379, 420)] ],

                  'DATA8.TXT': [ [(129, 160)], [(205, 254)], [(266, 318)], [(332, 381)], [(392, 435)], [(448, 486)] ],

                  'DATA9.TXT': [ [(60, 79)], [(92, 139)], [(150, 200)], [(216, 224)], [(247, 292)], [(302, 351)] ],

                  'DATA10.TXT': [ [(98, 122)], [(160, 211)], [(228, 261)], [(272, 326)], [(352, 424)] ],

                  'DATA11.TXT': [ [(45, 62)], [(94, 142)], [(152, 203)], [(214, 262)], [(273, 318)], [(332, 382)] ]

                 }



def process_data_file(data_basename, md):

    data_filename = '/kaggle/input/airlogger/' + data_basename

    print(data_basename)

    print(md.loc[data_basename]['description'])



    df = pd.read_csv(data_filename, sep=";")

    # Insert time column. We did one measurement per 2 seconds.

    df.insert(0, 'Time_s', df.index * 2)



    fig = go.Figure(go.Scattermapbox(

        mode = "markers",

        lon = [md.loc[data_basename]['lon']],

        lat = [md.loc[data_basename]['lat']],

        marker = {'size': 15}))

    fig.update_layout(

        mapbox_style="open-street-map",

        mapbox = {'zoom': 13, 'center': {'lon': md.loc[data_basename]['lon'], 'lat': md.loc[data_basename]['lat']}}

    )

    fig.show()



    fig_profile = make_subplots(specs=[[{"secondary_y": True}]])

    fig_profile.update_layout(title='Flight profile, ' + md.loc[data_basename]['datetime'] + ', ' + md.loc[data_basename]['location'])

    fig_profile.update_xaxes(title='Time, s')

    fig_profile.update_yaxes(title='PM, ug/m3', secondary_y=False)

    fig_profile.update_yaxes(title='Altitude, m', secondary_y=True)

    fig_profile.add_trace(go.Scatter(x=df['Time_s'], y=df['PM10_ug/m3'], name='PM 10', line=dict(color='red')))

    fig_profile.add_trace(go.Scatter(x=df['Time_s'], y=df['PM2.5_ug/m3'], name='PM 2.5', line=dict(color='orange')))

    fig_profile.add_trace(go.Scatter(x=df['Time_s'], y=df['PM1_ug/m3'], name='PM 1', line=dict(color='lime')))

    fig_profile.add_trace(go.Scatter(x=df['Time_s'], y=df['Altitude_m'], name='Altitude', line=dict(color='black')), secondary_y=True)

    fig_profile.show()



    avg_series = list()



    with open('stat_' + data_basename, 'w') as statf:

        for shelf in range(len(shelves_ranges[data_basename])):

            # Merge several series of measurements into one

            rlist = [df.iloc[shelves_ranges[data_basename][shelf][r][0]:shelves_ranges[data_basename][shelf][r][1]] for r in range(len(shelves_ranges[data_basename][shelf]))]

            piece = pd.concat(rlist)

            # Remove time column from stat dataframe

            piece.pop('Time_s')

            # Write statistics for each set of measurements

            statf.write('Mean\n')

            statf.write(piece.mean().to_string() + '\n')

            statf.write('StdDev\n')

            statf.write(piece.std().to_string() + '\n')

            statf.write('StdError\n')

            statf.write(pd.Series(piece.std() / piece.count()**0.5).to_string() + '\n')

            statf.write('\n')

            avg_series.append(piece.mean())



    # Pivot table to display statistics

    pivot = pd.DataFrame(avg_series, columns=avg_series[0].index)

    pivot.head(11)

    

    # Draw a plot



    fig_second = make_subplots()

    fig_second.update_layout(title='PM ' + md.loc[data_basename]['datetime'] + ', ' + md.loc[data_basename]['location'])

    fig_second.update_xaxes(title='Altitude, m')

    fig_second.update_yaxes(title='PM, ug/m3')

    fig_second.add_trace(go.Scatter(x=pivot['Altitude_m'], y=pivot['PM10_ug/m3'], name='PM 10', line=dict(color='red')))

    fig_second.add_trace(go.Scatter(x=pivot['Altitude_m'], y=pivot['PM2.5_ug/m3'], name='PM 2.5', line=dict(color='orange')))

    fig_second.add_trace(go.Scatter(x=pivot['Altitude_m'], y=pivot['PM1_ug/m3'], name='PM 1', line=dict(color='lime')))

    fig_second.show()

    

    fig_third = make_subplots(specs=[[{"secondary_y": True}]])

    fig_third.update_layout(title='Temperature and humidity ' + md.loc[data_basename]['datetime'] + ', ' + md.loc[data_basename]['location'])

    fig_third.update_xaxes(title='Altitude, m')

    fig_third.update_yaxes(title='Temperature, degC')

    fig_third.update_yaxes(title='Humidity, %', secondary_y=True)

    fig_third.add_trace(go.Scatter(x=pivot['Altitude_m'], y=pivot['Temperature_degC'], name='Temperature', line=dict(color='blue')))

    fig_third.add_trace(go.Scatter(x=pivot['Altitude_m'], y=pivot['Humidity_%'], name='Humidity', line=dict(color='lightblue')), secondary_y=True)

    fig_third.show()

metadata_filename = '/kaggle/input/airlogger/airlogger_metadata.txt'

md = pd.read_csv(metadata_filename, sep=';', index_col=0)



fig_map = px.scatter_mapbox(md, lon='lon', lat='lat', hover_name='location', text='datetime', zoom=10)

fig_map.update_layout(mapbox_style="open-street-map")

fig_map.show()
for data_basename in shelves_ranges:

    process_data_file(data_basename, md)